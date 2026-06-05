package main

import (
	"log"
	"net"
	"os"
	"strings"
	"sync/atomic"

	"golang.org/x/sys/unix"
)

func main() {
	listenAddr := envOrDefault("LB_ADDR", "0.0.0.0:9999")
	upstreamsStr := envOrDefault("LB_UPSTREAMS", "/sockets/api1.sock,/sockets/api2.sock")
	upstreams := strings.Split(upstreamsStr, ",")

	l, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	defer l.Close()

	log.Printf("LB listening on %s, upstreams: %v", listenAddr, upstreams)

	var next uint32
	for {
		conn, err := l.Accept()
		if err != nil {
			log.Printf("accept error: %v", err)
			continue
		}

		// Fast handoff
		go func(c net.Conn) {
			defer c.Close()

			tcpConn, ok := c.(*net.TCPConn)
			if !ok {
				return
			}

			// Minimize latency
			_ = tcpConn.SetNoDelay(true)

			// We need the raw FD to send it via SCM_RIGHTS
			f, err := tcpConn.File()
			if err != nil {
				log.Printf("failed to get file descriptor: %v", err)
				return
			}
			defer f.Close()

			fd := int(f.Fd())

			idx := atomic.AddUint32(&next, 1) % uint32(len(upstreams))
			target := upstreams[idx]

			if err := handoff(target, fd); err != nil {
				log.Printf("handoff to %s failed: %v", target, err)
			}
		}(conn)
	}
}

func handoff(socketPath string, clientFd int) error {
	// Connect to the upstream Unix socket
	raddr, err := net.ResolveUnixAddr("unix", socketPath)
	if err != nil {
		return err
	}

	conn, err := net.DialUnix("unix", nil, raddr)
	if err != nil {
		return err
	}
	defer conn.Close()

	// Get the FD of the unix connection to use unix.SendMsg
	unixFile, err := conn.File()
	if err != nil {
		return err
	}
	defer unixFile.Close()

	unixFd := int(unixFile.Fd())

	// SCM_RIGHTS payload
	// We send 8 bytes of dummy data (could be timestamp like r-backend-26)
	dummy := make([]byte, 8)
	oob := unix.UnixRights(clientFd)

	return unix.Sendmsg(unixFd, dummy, oob, nil, 0)

}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
