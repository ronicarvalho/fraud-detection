package main

import (
	"math/rand"
	"testing"
)

func BenchmarkFraudCountTop5(b *testing.B) {
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
			queries[i][j] = int8(rng.Intn(255) - 127)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ds.FraudCountTop5(&queries[i%len(queries)])
	}
}
