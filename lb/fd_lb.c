// fd_lb — minimal TCP→Unix-domain fd-passing load balancer.
//
// Listens on a TCP port, accepts incoming connections, and hands the raw
// client file descriptor over to one of two backend processes via
// SCM_RIGHTS on a persistent Unix-domain control socket. The backend then
// reads/writes that fd directly to the client — the LB no longer sits in
// the data path. Round-robin between the two backends.
//
// Env:
//   FD_LB_ADDR      listen address (default 0.0.0.0:9999)
//   FD_TARGET_1     unix socket path to backend 1   (required)
//   FD_TARGET_2     unix socket path to backend 2   (required)
//   FD_LB_CONNECT_RETRIES    initial connect retries per backend (default 3000)
//   FD_LB_CONNECT_SLEEP_MS   sleep between retries  (default 10)

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define BACKLOG 4096

static int g_connect_retries = 3000;
static int g_connect_sleep_ms = 10;

typedef struct {
    const char *path;
    int ctrl_fd;
} backend_t;

static int env_int(const char *name, int fallback, int min, int max) {
    const char *v = getenv(name);
    if (!v || !*v) return fallback;
    char *end = NULL;
    long n = strtol(v, &end, 10);
    if (end == v || n < min || n > max) return fallback;
    return (int)n;
}

static int create_tcp_listener(const char *addr) {
    char host[64] = "0.0.0.0";
    int port = 9999;
    if (addr) {
        const char *colon = strrchr(addr, ':');
        if (colon && colon != addr) {
            size_t hlen = (size_t)(colon - addr);
            if (hlen >= sizeof(host)) hlen = sizeof(host) - 1;
            memcpy(host, addr, hlen);
            host[hlen] = '\0';
            port = atoi(colon + 1);
        } else if (colon) {
            port = atoi(colon + 1);
        }
    }

    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) { perror("socket"); return -1; }

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        fprintf(stderr, "invalid host: %s\n", host);
        close(fd);
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        perror("bind"); close(fd); return -1;
    }
    if (listen(fd, BACKLOG) < 0) {
        perror("listen"); close(fd); return -1;
    }
    fprintf(stderr, "fd_lb listening on %s:%d\n", host, port);
    return fd;
}

static int connect_unix(const char *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof(sa));
    sa.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof(sa.sun_path)) { close(fd); return -1; }
    strncpy(sa.sun_path, path, sizeof(sa.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int connect_backend_retry(backend_t *b, int retries) {
    for (int i = 0; i < retries; ++i) {
        b->ctrl_fd = connect_unix(b->path);
        if (b->ctrl_fd >= 0) return 0;
        if (g_connect_sleep_ms > 0) {
            struct timespec ts;
            ts.tv_sec = g_connect_sleep_ms / 1000;
            ts.tv_nsec = (g_connect_sleep_ms % 1000) * 1000L * 1000L;
            nanosleep(&ts, NULL);
        }
    }
    return -1;
}

// Send file descriptor `fd_to_send` over `sock` using SCM_RIGHTS. The kernel
// duplicates the fd into the receiving process — the receiver gets its own
// reference to the same socket, and can read/write directly to the client.
static int send_passed_fd(int sock, int fd_to_send) {
    char dummy = 0;
    struct iovec iov;
    iov.iov_base = &dummy;
    iov.iov_len = 1;

    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } u;
    memset(&u, 0, sizeof(u));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = u.buf;
    msg.msg_controllen = sizeof(u.buf);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));

    for (;;) {
        ssize_t n = sendmsg(sock, &msg, MSG_NOSIGNAL);
        if (n == 1) return 0;
        if (n < 0 && errno == EINTR) continue;
        return -1;
    }
}

static int forward(backend_t *b, int cfd) {
    if (send_passed_fd(b->ctrl_fd, cfd) == 0) return 0;
    // Control socket broke — try one reconnect + retry
    close(b->ctrl_fd);
    b->ctrl_fd = -1;
    if (connect_backend_retry(b, 3) < 0) return -1;
    return send_passed_fd(b->ctrl_fd, cfd);
}

int main(void) {
    const char *t1 = getenv("FD_TARGET_1");
    const char *t2 = getenv("FD_TARGET_2");
    if (!t1 || !*t1 || !t2 || !*t2) {
        fprintf(stderr, "fd_lb: FD_TARGET_1 and FD_TARGET_2 are required\n");
        return 1;
    }

    g_connect_retries = env_int("FD_LB_CONNECT_RETRIES", 3000, 1, 30000);
    g_connect_sleep_ms = env_int("FD_LB_CONNECT_SLEEP_MS", 10, 0, 1000);

    int listen_fd = create_tcp_listener(getenv("FD_LB_ADDR"));
    if (listen_fd < 0) return 1;

    backend_t backends[2] = {{t1, -1}, {t2, -1}};
    if (connect_backend_retry(&backends[0], g_connect_retries) < 0 ||
        connect_backend_retry(&backends[1], g_connect_retries) < 0) {
        fprintf(stderr, "fd_lb: failed to connect to backends\n");
        return 1;
    }
    fprintf(stderr, "fd_lb: connected to %s and %s\n", t1, t2);

    uint64_t rr = 0;
    int one = 1;
    for (;;) {
        int cfd = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        backend_t *b = &backends[rr++ & 1];
        if (forward(b, cfd) < 0) {
            close(cfd);
            continue;
        }
        close(cfd);
    }

    close(listen_fd);
    if (backends[0].ctrl_fd >= 0) close(backends[0].ctrl_fd);
    if (backends[1].ctrl_fd >= 0) close(backends[1].ctrl_fd);
    return 1;
}
