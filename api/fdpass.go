package main

import (
	"errors"
	"fmt"
	"net"
	"os"
	"sync"
	"syscall"
)

// fdPassListener implements net.Listener on top of an SCM_RIGHTS control
// channel. The LB connects once to a Unix-domain socket we listen on and
// then sends one byte + an attached file descriptor per inbound TCP
// connection. Each fd we receive is wrapped as a net.Conn and queued on
// the accept channel — from fasthttp's point of view, it's a regular
// listener handing out TCP connections.
type fdPassListener struct {
	ctrlLn    *net.UnixListener
	addr      net.Addr
	accepted  chan net.Conn
	errc      chan error
	closeOnce sync.Once
	done      chan struct{}
}

func newFDPassListener(path string) (*fdPassListener, error) {
	_ = os.Remove(path)

	ln, err := net.Listen("unix", path)
	if err != nil {
		return nil, fmt.Errorf("listen unix %s: %w", path, err)
	}
	if err := os.Chmod(path, 0o666); err != nil {
		ln.Close()
		return nil, fmt.Errorf("chmod %s: %w", path, err)
	}

	ul, ok := ln.(*net.UnixListener)
	if !ok {
		ln.Close()
		return nil, errors.New("unexpected listener type")
	}

	l := &fdPassListener{
		ctrlLn:   ul,
		addr:     &net.TCPAddr{Port: 0}, // not meaningful — fasthttp only logs it
		accepted: make(chan net.Conn, 1024),
		errc:     make(chan error, 1),
		done:     make(chan struct{}),
	}
	go l.acceptControlLoop()
	return l, nil
}

func (l *fdPassListener) acceptControlLoop() {
	for {
		c, err := l.ctrlLn.AcceptUnix()
		if err != nil {
			select {
			case <-l.done:
			default:
				l.errc <- err
			}
			return
		}
		go l.recvLoop(c)
	}
}

func (l *fdPassListener) recvLoop(c *net.UnixConn) {
	defer c.Close()

	oob := make([]byte, syscall.CmsgSpace(4))
	buf := make([]byte, 1)

	for {
		_, oobn, _, _, err := c.ReadMsgUnix(buf, oob)
		if err != nil {
			return
		}
		if oobn == 0 {
			continue
		}
		conn, err := connFromCmsg(oob[:oobn])
		if err != nil {
			continue
		}
		select {
		case l.accepted <- conn:
		case <-l.done:
			conn.Close()
			return
		}
	}
}

// connFromCmsg parses the SCM_RIGHTS control message, takes the first fd,
// and wraps it as a net.Conn. net.FileConn duplicates the fd internally,
// so we close the os.File without losing the underlying socket.
func connFromCmsg(oob []byte) (net.Conn, error) {
	scms, err := syscall.ParseSocketControlMessage(oob)
	if err != nil {
		return nil, err
	}
	for _, scm := range scms {
		fds, err := syscall.ParseUnixRights(&scm)
		if err != nil || len(fds) == 0 {
			continue
		}
		// First fd is the passed connection. Close any extras.
		for _, extra := range fds[1:] {
			syscall.Close(extra)
		}
		fd := fds[0]
		syscall.CloseOnExec(fd)
		f := os.NewFile(uintptr(fd), "fdpass")
		if f == nil {
			syscall.Close(fd)
			return nil, errors.New("os.NewFile returned nil")
		}
		conn, err := net.FileConn(f)
		f.Close() // FileConn dup'd the fd
		if err != nil {
			return nil, err
		}
		return conn, nil
	}
	return nil, errors.New("no fd in cmsg")
}

func (l *fdPassListener) Accept() (net.Conn, error) {
	select {
	case c := <-l.accepted:
		return c, nil
	case err := <-l.errc:
		return nil, err
	case <-l.done:
		return nil, net.ErrClosed
	}
}

func (l *fdPassListener) Close() error {
	l.closeOnce.Do(func() {
		close(l.done)
		l.ctrlLn.Close()
		drained := true
		for drained {
			select {
			case c := <-l.accepted:
				c.Close()
			default:
				drained = false
			}
		}
	})
	return nil
}

func (l *fdPassListener) Addr() net.Addr { return l.addr }
