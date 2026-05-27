//go:build !amd64

package main

import "unsafe"

func dist8(q *Vector, blockPtr unsafe.Pointer, out *[BlockSize]float32) {
	dist8Scalar(q, blockPtr, out)
}
