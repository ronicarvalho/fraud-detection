package main

import (
	"math/rand"
	"testing"
)

func BenchmarkFraudScore(b *testing.B) {
	ds, err := LoadDataset("/tmp/fraud-data/references.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer ds.Close()
	b.Logf("dataset: %d entries", ds.Len())

	rng := rand.New(rand.NewSource(1))
	queries := make([]Vector, 64)
	for i := range queries {
		for j := 0; j < 14; j++ {
			queries[i][j] = int16(rng.Intn(2*Scale+1) - Scale)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ds.FraudScore(&queries[i%len(queries)])
	}
}
