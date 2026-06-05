package main

import (
	"log"
	"net"
	"os"
	"syscall"
	"time"

	"github.com/valyala/fasthttp"
	"golang.org/x/sys/unix"
)

var (
	ds  *Dataset
	cfg *Config
)

func main() {
	dataDir := envOrDefault("DATA_DIR", "/app/data")

	var err error
	cfg, err = loadConfig(dataDir+"/normalization.json", dataDir+"/mcc_risk.json")
	if err != nil {
		log.Fatalf("config load: %v", err)
	}
	log.Printf("config loaded (%d mcc entries)", len(cfg.MccRisk))

	ds, err = LoadDataset(dataDir + "/references.bin")
	if err != nil {
		log.Fatalf("dataset load: %v", err)
	}
	defer ds.Close()
	log.Printf("dataset loaded (%d entries)", ds.Len())

	instanceID := envOrDefault("INSTANCE_ID", "?")
	socketPath := os.Getenv("RINHA_SOCKET")

	server := &fasthttp.Server{
		Handler:            handler,
		Name:               "fraud-detector-api",
		MaxConnsPerIP:      4096,
		MaxRequestBodySize: 64 * 1024,
		ReadTimeout:        5 * time.Second,
		WriteTimeout:       5 * time.Second,
		TCPKeepalive:       true,
	}

	if socketPath != "" {
		log.Printf("api instance %s listening on unix:%s (fd-handoff mode)", instanceID, socketPath)
		_ = os.Remove(socketPath)
		l, err := net.Listen("unix", socketPath)
		if err != nil {
			log.Fatalf("unix listen: %v", err)
		}
		
		ln := &fdListener{unixL: l.(*net.UnixListener)}
		if err := server.Serve(ln); err != nil {
			log.Fatalf("server: %v", err)
		}
	} else {
		log.Printf("api instance %s listening on :8080", instanceID)
		if err := server.ListenAndServe(":8080"); err != nil {
			log.Fatalf("server: %v", err)
		}
	}
}

type fdListener struct {
	unixL *net.UnixListener
}

func (l *fdListener) Accept() (net.Conn, error) {
	for {
		unixConn, err := l.unixL.AcceptUnix()
		if err != nil {
			return nil, err
		}

		// Receive the FD via SCM_RIGHTS
		f, err := unixConn.File()
		if err != nil {
			unixConn.Close()
			continue
		}

		buf := make([]byte, 8)
		oob := make([]byte, 32)
		_, oobn, _, _, err := unix.Recvmsg(int(f.Fd()), buf, oob, 0)
		f.Close()
		unixConn.Close()

		if err != nil {
			continue
		}

		msgs, err := unix.ParseSocketControlMessage(oob[:oobn])
		if err != nil || len(msgs) == 0 {
			continue
		}

		fds, err := unix.ParseUnixRights(&msgs[0])
		if err != nil || len(fds) == 0 {
			continue
		}

		clientFd := fds[0]
		// Important: set non-blocking for fasthttp/go-network-stack
		if err := syscall.SetNonblock(clientFd, true); err != nil {
			unix.Close(clientFd)
			continue
		}

		clientFile := os.NewFile(uintptr(clientFd), "client-tcp")
		clientConn, err := net.FileConn(clientFile)
		clientFile.Close() // net.FileConn duplicates the FD, so we can close this one

		if err != nil {
			continue
		}
		return clientConn, nil
	}
}

func (l *fdListener) Close() error {
	return l.unixL.Close()
}

func (l *fdListener) Addr() net.Addr {
	return l.unixL.Addr()
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
