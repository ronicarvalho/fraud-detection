//go:build amd64

package main

import "unsafe"

// dist8AVX2 computes squared euclidean distances between the query and 8 vectors
// stored in pair-SoA layout (224 bytes), returning 8 float32 in `out`.
// Implemented in dist8_amd64.s using AVX2 (VPMADDWD + VCVTDQ2PS + VADDPS).
//
//go:noescape
func dist8AVX2(q *Vector, blockPtr unsafe.Pointer, out *[BlockSize]float32)

// dist8 is the dispatcher used by the hot scan loop.
// Fase 1 baseline: usa o escalar (auto-vetorizado pelo Go onde possível).
// Em produção a chamada repetida da função assembly dist8AVX2 introduziu
// overhead que anulou o ganho do SIMD em microbench. Mantemos a função
// dist8AVX2 disponível para validação e benchmarks; o switch real acontece
// em scanCluster.
func dist8(q *Vector, blockPtr unsafe.Pointer, out *[BlockSize]float32) {
	dist8Scalar(q, blockPtr, out)
}
