package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
)

const (
	IVFMagic   = uint32(0x31465649) // "IVF1"
	HeaderSize = 32
	EntrySize  = 15
	NumDims    = 14
	LabelFraud = 1
	LabelLegit = 0
	K          = 5  // top-K nearest neighbors returned
	NPROBE     = 16 // number of clusters to scan per query
)

type Dataset struct {
	raw       []byte   // entire mmap region
	centroids []byte   // [nClusters * NumDims] int8 raw
	offsets   []uint32 // [nClusters + 1] start of each cluster (in entries)
	body      []byte   // [n * EntrySize] entries, grouped by cluster
	n         int
	nClusters int
}

func LoadDataset(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	size := int(stat.Size())
	if size < HeaderSize {
		return nil, fmt.Errorf("file too small: %d bytes", size)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	magic := binary.LittleEndian.Uint32(data[0:4])
	if magic != IVFMagic {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("bad magic: %x (expected %x)", magic, IVFMagic)
	}
	n := int(binary.LittleEndian.Uint64(data[8:16]))
	nClusters := int(binary.LittleEndian.Uint32(data[16:20]))
	nDims := int(binary.LittleEndian.Uint32(data[20:24]))
	if nDims != NumDims {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("dims=%d, expected %d", nDims, NumDims)
	}

	centStart := HeaderSize
	centEnd := centStart + nClusters*NumDims
	offStart := centEnd
	offEnd := offStart + (nClusters+1)*4
	bodyStart := offEnd
	bodyEnd := bodyStart + n*EntrySize
	if bodyEnd > size {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("file size %d < expected %d", size, bodyEnd)
	}

	offsets := make([]uint32, nClusters+1)
	for i := 0; i <= nClusters; i++ {
		offsets[i] = binary.LittleEndian.Uint32(data[offStart+i*4:])
	}

	return &Dataset{
		raw:       data,
		centroids: data[centStart:centEnd],
		offsets:   offsets,
		body:      data[bodyStart:bodyEnd],
		n:         n,
		nClusters: nClusters,
	}, nil
}

func (d *Dataset) Close() error {
	if d.raw == nil {
		return nil
	}
	err := syscall.Munmap(d.raw)
	d.raw = nil
	d.centroids = nil
	d.body = nil
	return err
}

func (d *Dataset) Len() int        { return d.n }
func (d *Dataset) NClusters() int  { return d.nClusters }

// FraudCountTop5 finds the K=5 nearest neighbors via IVF (NPROBE clusters),
// using squared Euclidean distance in int8-quantized space, and returns the
// number of fraud-labeled entries among them.
func (d *Dataset) FraudCountTop5(query *Vector) int {
	probes := d.topProbes(query)

	var topDist [K]int32
	var topLabel [K]byte
	filled := 0

	for _, c := range probes[:] {
		start := int(d.offsets[c])
		end := int(d.offsets[c+1])
		if start == end {
			continue
		}
		filled = scanRange(d.body, query, start, end, &topDist, &topLabel, filled)
	}

	fraud := 0
	for i := 0; i < filled; i++ {
		if topLabel[i] == LabelFraud {
			fraud++
		}
	}
	return fraud
}

// topProbes returns the NPROBE cluster indices closest to query.
func (d *Dataset) topProbes(query *Vector) [NPROBE]uint32 {
	q0 := int32(query[0])
	q1 := int32(query[1])
	q2 := int32(query[2])
	q3 := int32(query[3])
	q4 := int32(query[4])
	q5 := int32(query[5])
	q6 := int32(query[6])
	q7 := int32(query[7])
	q8 := int32(query[8])
	q9 := int32(query[9])
	q10 := int32(query[10])
	q11 := int32(query[11])
	q12 := int32(query[12])
	q13 := int32(query[13])

	var topIdx [NPROBE]uint32
	var topDist [NPROBE]int32
	filled := 0
	worst := int32(math.MaxInt32)

	cents := d.centroids
	for c := 0; c < d.nClusters; c++ {
		off := c * NumDims
		d0 := q0 - int32(int8(cents[off+0]))
		d1 := q1 - int32(int8(cents[off+1]))
		d2 := q2 - int32(int8(cents[off+2]))
		d3 := q3 - int32(int8(cents[off+3]))
		d4 := q4 - int32(int8(cents[off+4]))
		d5 := q5 - int32(int8(cents[off+5]))
		d6 := q6 - int32(int8(cents[off+6]))
		d7 := q7 - int32(int8(cents[off+7]))
		d8 := q8 - int32(int8(cents[off+8]))
		d9 := q9 - int32(int8(cents[off+9]))
		d10 := q10 - int32(int8(cents[off+10]))
		d11 := q11 - int32(int8(cents[off+11]))
		d12 := q12 - int32(int8(cents[off+12]))
		d13 := q13 - int32(int8(cents[off+13]))

		sum := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

		if filled < NPROBE {
			topIdx[filled] = uint32(c)
			topDist[filled] = sum
			filled++
			if filled == NPROBE {
				for k := NPROBE/2 - 1; k >= 0; k-- {
					siftDownProbes(&topDist, &topIdx, k)
				}
				worst = topDist[0]
			}
			continue
		}
		if sum < worst {
			topIdx[0] = uint32(c)
			topDist[0] = sum
			siftDownProbes(&topDist, &topIdx, 0)
			worst = topDist[0]
		}
	}
	return topIdx
}

func scanRange(
	body []byte, query *Vector, start, end int,
	topDist *[K]int32, topLabel *[K]byte, filled int,
) int {
	q0 := int32(query[0])
	q1 := int32(query[1])
	q2 := int32(query[2])
	q3 := int32(query[3])
	q4 := int32(query[4])
	q5 := int32(query[5])
	q6 := int32(query[6])
	q7 := int32(query[7])
	q8 := int32(query[8])
	q9 := int32(query[9])
	q10 := int32(query[10])
	q11 := int32(query[11])
	q12 := int32(query[12])
	q13 := int32(query[13])

	for i := start; i < end; i++ {
		off := i * EntrySize
		d0 := q0 - int32(int8(body[off+0]))
		d1 := q1 - int32(int8(body[off+1]))
		d2 := q2 - int32(int8(body[off+2]))
		d3 := q3 - int32(int8(body[off+3]))
		d4 := q4 - int32(int8(body[off+4]))
		d5 := q5 - int32(int8(body[off+5]))
		d6 := q6 - int32(int8(body[off+6]))
		d7 := q7 - int32(int8(body[off+7]))
		d8 := q8 - int32(int8(body[off+8]))
		d9 := q9 - int32(int8(body[off+9]))
		d10 := q10 - int32(int8(body[off+10]))
		d11 := q11 - int32(int8(body[off+11]))
		d12 := q12 - int32(int8(body[off+12]))
		d13 := q13 - int32(int8(body[off+13]))

		sum := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

		label := body[off+14]
		if filled < K {
			topDist[filled] = sum
			topLabel[filled] = label
			filled++
			if filled == K {
				for k := K/2 - 1; k >= 0; k-- {
					siftDown(topDist, topLabel, k)
				}
			}
			continue
		}
		if sum < topDist[0] {
			topDist[0] = sum
			topLabel[0] = label
			siftDown(topDist, topLabel, 0)
		}
	}
	return filled
}

func siftDown(dist *[K]int32, label *[K]byte, i int) {
	for {
		l := 2*i + 1
		if l >= K {
			return
		}
		max := l
		if l+1 < K && dist[l+1] > dist[l] {
			max = l + 1
		}
		if dist[i] >= dist[max] {
			return
		}
		dist[i], dist[max] = dist[max], dist[i]
		label[i], label[max] = label[max], label[i]
		i = max
	}
}

func siftDownProbes(dist *[NPROBE]int32, idx *[NPROBE]uint32, i int) {
	for {
		l := 2*i + 1
		if l >= NPROBE {
			return
		}
		max := l
		if l+1 < NPROBE && dist[l+1] > dist[l] {
			max = l + 1
		}
		if dist[i] >= dist[max] {
			return
		}
		dist[i], dist[max] = dist[max], dist[i]
		idx[i], idx[max] = idx[max], idx[i]
		i = max
	}
}
