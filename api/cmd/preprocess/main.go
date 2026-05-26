package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"
)

const (
	IVFMagic   = uint32(0x33465649) // "IVF3"
	HeaderSize = 32
	NumDims    = 14
	DimBytes   = NumDims * 2  // 28
	EntrySize  = DimBytes + 2 // 30 (28 vector + 1 label + 1 pad)
	Scale      = 10000
)

type refEntry struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

func main() {
	synthN := flag.Int("synth", 0, "if >0, generate N synthetic entries instead of reading input")
	k := flag.Int("k", 2048, "number of IVF clusters")
	iters := flag.Int("iters", 60, "k-means iterations")
	batchSize := flag.Int("batch", 8192, "mini-batch size per iteration")
	flag.Parse()

	args := flag.Args()
	var outPath string
	var entries [][NumDims]int16
	var labels []byte

	switch {
	case *synthN > 0:
		if len(args) != 1 {
			log.Fatalf("usage: preprocess -synth N <output.bin>")
		}
		outPath = args[0]
		entries, labels = genSynth(*synthN)
	default:
		if len(args) != 2 {
			log.Fatalf("usage: preprocess <references.json.gz> <output.bin>")
		}
		outPath = args[1]
		var err error
		entries, labels, err = loadGzip(args[0])
		if err != nil {
			log.Fatal(err)
		}
	}

	log.Printf("loaded %d entries", len(entries))

	if *k > len(entries) {
		*k = len(entries)
	}

	t0 := time.Now()
	centroids := trainKMeans(entries, *k, *iters, *batchSize)
	log.Printf("trained %d centroids in %v", *k, time.Since(t0))

	t1 := time.Now()
	assignments := assignAll(entries, centroids)
	log.Printf("assigned %d entries in %v", len(entries), time.Since(t1))

	t2 := time.Now()
	bboxMin, bboxMax := computeBboxes(entries, assignments, *k)
	log.Printf("computed bboxes in %v", time.Since(t2))

	if err := writeIVF(outPath, centroids, bboxMin, bboxMax, entries, labels, assignments); err != nil {
		log.Fatal(err)
	}
}

func loadGzip(path string) ([][NumDims]int16, []byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, err
	}
	defer gr.Close()

	dec := json.NewDecoder(gr)
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, nil, fmt.Errorf("expected '[', got %v", tok)
	}

	entries := make([][NumDims]int16, 0, 3_100_000)
	labels := make([]byte, 0, 3_100_000)
	var n int
	for dec.More() {
		var e refEntry
		if err := dec.Decode(&e); err != nil {
			return nil, nil, fmt.Errorf("decode entry %d: %w", n, err)
		}
		if len(e.Vector) != NumDims {
			return nil, nil, fmt.Errorf("entry %d: %d dims, want %d", n, len(e.Vector), NumDims)
		}
		var v [NumDims]int16
		for j := 0; j < NumDims; j++ {
			v[j] = quantizeRef(e.Vector[j])
		}
		entries = append(entries, v)
		labels = append(labels, labelByte(e.Label))
		n++
		if n%500_000 == 0 {
			log.Printf("read %d entries", n)
		}
	}
	return entries, labels, nil
}

func genSynth(n int) ([][NumDims]int16, []byte) {
	rng := rand.New(rand.NewSource(42))
	entries := make([][NumDims]int16, n)
	labels := make([]byte, n)
	for i := 0; i < n; i++ {
		for j := 0; j < NumDims; j++ {
			entries[i][j] = quantizeRef(rng.Float32())
		}
		if rng.Float32() < 0.2 {
			labels[i] = 1
		}
	}
	return entries, labels
}

func trainKMeans(data [][NumDims]int16, k, iters, batchSize int) [][NumDims]int16 {
	rng := rand.New(rand.NewSource(1))

	centroids := make([][NumDims]float32, k)
	for c := 0; c < k; c++ {
		idx := rng.Intn(len(data))
		for j := 0; j < NumDims; j++ {
			centroids[c][j] = float32(data[idx][j])
		}
	}
	counts := make([]int, k)

	for it := 0; it < iters; it++ {
		for b := 0; b < batchSize; b++ {
			x := &data[rng.Intn(len(data))]
			best := nearestF32(x, centroids)
			counts[best]++
			lr := 1.0 / float32(counts[best])
			for j := 0; j < NumDims; j++ {
				centroids[best][j] = centroids[best][j]*(1-lr) + float32(x[j])*lr
			}
		}
		if it%5 == 0 || it == iters-1 {
			log.Printf("k-means iter %d/%d", it+1, iters)
		}
	}

	out := make([][NumDims]int16, k)
	for c := 0; c < k; c++ {
		for j := 0; j < NumDims; j++ {
			v := centroids[c][j]
			if v < -Scale {
				out[c][j] = -Scale
			} else if v > Scale {
				out[c][j] = Scale
			} else {
				out[c][j] = int16(math.Round(float64(v)))
			}
		}
	}
	return out
}

func nearestF32(x *[NumDims]int16, centroids [][NumDims]float32) int {
	bestIdx := 0
	bestDist := float32(math.MaxFloat32)
	for c := 0; c < len(centroids); c++ {
		var sum float32
		for j := 0; j < NumDims; j++ {
			d := centroids[c][j] - float32(x[j])
			sum += d * d
		}
		if sum < bestDist {
			bestDist = sum
			bestIdx = c
		}
	}
	return bestIdx
}

func nearestI16(x *[NumDims]int16, centroids [][NumDims]int16) int {
	bestIdx := 0
	bestDist := int64(math.MaxInt64)
	for c := 0; c < len(centroids); c++ {
		var sum int64
		for j := 0; j < NumDims; j++ {
			d := int64(centroids[c][j]) - int64(x[j])
			sum += d * d
		}
		if sum < bestDist {
			bestDist = sum
			bestIdx = c
		}
	}
	return bestIdx
}

func assignAll(data [][NumDims]int16, centroids [][NumDims]int16) []uint32 {
	n := len(data)
	out := make([]uint32, n)
	workers := runtime.NumCPU()
	if workers > 32 {
		workers = 32
	}
	chunk := (n + workers - 1) / workers

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > n {
			end = n
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				out[i] = uint32(nearestI16(&data[i], centroids))
			}
		}(start, end)
	}
	wg.Wait()
	return out
}

// computeBboxes scans every entry and records per-cluster min/max for each dim.
// Clusters with no entries get a degenerate bbox of all-zero (lb will be ~q²).
func computeBboxes(entries [][NumDims]int16, assignments []uint32, k int) ([][NumDims]int16, [][NumDims]int16) {
	bMin := make([][NumDims]int16, k)
	bMax := make([][NumDims]int16, k)
	seen := make([]bool, k)
	for i, e := range entries {
		c := assignments[i]
		if !seen[c] {
			bMin[c] = e
			bMax[c] = e
			seen[c] = true
			continue
		}
		for j := 0; j < NumDims; j++ {
			if e[j] < bMin[c][j] {
				bMin[c][j] = e[j]
			}
			if e[j] > bMax[c][j] {
				bMax[c][j] = e[j]
			}
		}
	}
	return bMin, bMax
}

func writeIVF(
	path string,
	centroids [][NumDims]int16,
	bboxMin, bboxMax [][NumDims]int16,
	entries [][NumDims]int16,
	labels []byte,
	assignments []uint32,
) error {
	nC := len(centroids)
	n := len(entries)

	clusterCounts := make([]uint32, nC)
	for _, a := range assignments {
		clusterCounts[a]++
	}
	offsets := make([]uint32, nC+1)
	for i := 0; i < nC; i++ {
		offsets[i+1] = offsets[i] + clusterCounts[i]
	}

	cursor := make([]uint32, nC)
	copy(cursor, offsets[:nC])

	body := make([]byte, n*EntrySize)
	for i := 0; i < n; i++ {
		c := assignments[i]
		pos := cursor[c]
		cursor[c]++
		off := int(pos) * EntrySize
		for j := 0; j < NumDims; j++ {
			binary.LittleEndian.PutUint16(body[off+j*2:], uint16(entries[i][j]))
		}
		body[off+DimBytes] = labels[i]
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var header [HeaderSize]byte
	binary.LittleEndian.PutUint32(header[0:4], IVFMagic)
	binary.LittleEndian.PutUint32(header[4:8], 3)
	binary.LittleEndian.PutUint64(header[8:16], uint64(n))
	binary.LittleEndian.PutUint32(header[16:20], uint32(nC))
	binary.LittleEndian.PutUint32(header[20:24], uint32(NumDims))
	if _, err := f.Write(header[:]); err != nil {
		return err
	}

	writeI16Block := func(vecs [][NumDims]int16) error {
		buf := make([]byte, nC*DimBytes)
		for c := 0; c < nC; c++ {
			for j := 0; j < NumDims; j++ {
				binary.LittleEndian.PutUint16(buf[c*DimBytes+j*2:], uint16(vecs[c][j]))
			}
		}
		_, err := f.Write(buf)
		return err
	}

	if err := writeI16Block(centroids); err != nil {
		return err
	}
	if err := writeI16Block(bboxMin); err != nil {
		return err
	}
	if err := writeI16Block(bboxMax); err != nil {
		return err
	}

	offBuf := make([]byte, (nC+1)*4)
	for i := 0; i <= nC; i++ {
		binary.LittleEndian.PutUint32(offBuf[i*4:], offsets[i])
	}
	if _, err := f.Write(offBuf); err != nil {
		return err
	}

	if _, err := f.Write(body); err != nil {
		return err
	}

	total := HeaderSize + 3*nC*DimBytes + (nC+1)*4 + n*EntrySize
	log.Printf("wrote %d entries / %d clusters (%d bytes)", n, nC, total)
	return nil
}

func labelByte(s string) byte {
	if s == "fraud" {
		return 1
	}
	return 0
}

func quantizeRef(v float32) int16 {
	if v <= -0.9999 {
		return -Scale
	}
	if v < 0 {
		return 0
	}
	if v > 1 {
		return Scale
	}
	x := v*Scale + 0.5
	return int16(x)
}
