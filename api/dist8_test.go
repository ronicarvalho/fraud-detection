package main

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"
)

// TestDist8AVX2MatchesScalar validates that the AVX2 implementation produces
// results equivalent to the scalar version (within float32 rounding tolerance,
// since accumulation orders differ — scalar does pairs 0..6 in order, AVX2 uses
// two parallel accumulators).
func TestDist8AVX2MatchesScalar(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("AVX2 path is amd64-only")
	}

	rng := rand.New(rand.NewSource(42))

	for trial := 0; trial < 1000; trial++ {
		var q Vector
		for i := 0; i < NumDims; i++ {
			q[i] = int16(rng.Intn(2*10000+1) - 10000)
		}

		var block [BlockVecSize]byte
		for i := 0; i < BlockVecSize; i += 2 {
			v := int16(rng.Intn(2*10000+1) - 10000)
			*(*int16)(unsafe.Pointer(&block[i])) = v
		}

		var expected, actual [BlockSize]float32
		dist8Scalar(&q, unsafe.Pointer(&block[0]), &expected)
		dist8AVX2(&q, unsafe.Pointer(&block[0]), &actual)

		// Maximum possible distance: 14 * 20000^2 = 5.6e9. float32 has 23 bits
		// of mantissa, so ULP at that scale is around 512. Generous tolerance.
		for i := 0; i < BlockSize; i++ {
			rel := math.Abs(float64(expected[i]-actual[i])) / math.Max(1, float64(expected[i]))
			if rel > 1e-5 {
				t.Fatalf("trial %d lane %d: scalar=%g avx2=%g rel=%g", trial, i, expected[i], actual[i], rel)
			}
		}
	}
}

func BenchmarkDist8Scalar(b *testing.B) {
	var q Vector
	var block [BlockVecSize]byte
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < NumDims; i++ {
		q[i] = int16(rng.Intn(2*10000+1) - 10000)
	}
	for i := 0; i < BlockVecSize; i += 2 {
		*(*int16)(unsafe.Pointer(&block[i])) = int16(rng.Intn(2*10000+1) - 10000)
	}
	var out [BlockSize]float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dist8Scalar(&q, unsafe.Pointer(&block[0]), &out)
	}
}

func BenchmarkDist8AVX2(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("AVX2 path is amd64-only")
	}
	var q Vector
	var block [BlockVecSize]byte
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < NumDims; i++ {
		q[i] = int16(rng.Intn(2*10000+1) - 10000)
	}
	for i := 0; i < BlockVecSize; i += 2 {
		*(*int16)(unsafe.Pointer(&block[i])) = int16(rng.Intn(2*10000+1) - 10000)
	}
	var out [BlockSize]float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dist8AVX2(&q, unsafe.Pointer(&block[0]), &out)
	}
}
