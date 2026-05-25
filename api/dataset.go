package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
	"unsafe"
)

const (
	IVFMagic   = uint32(0x32465649) // "IVF2" — int16 layout, breaks old IVF1 binaries
	HeaderSize = 32
	NumDims    = 14
	DimBytes   = NumDims * 2 // 28
	EntrySize  = DimBytes + 2 // 30: 14*int16 + 1 label + 1 pad (keeps int16 alignment per-entry)
	LabelFraud = 1
	LabelLegit = 0
	K          = 5
	NPROBE     = 16
)

type Dataset struct {
	raw       []byte   // entire mmap region
	centroids []byte   // [nClusters * DimBytes] int16 LE raw
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
	centEnd := centStart + nClusters*DimBytes
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

func (d *Dataset) Len() int       { return d.n }
func (d *Dataset) NClusters() int { return d.nClusters }

// FraudScore returns n_fraud_in_top5 / 5 (simple majority vote, matching the spec).
func (d *Dataset) FraudScore(query *Vector) float32 {
	probes := d.topProbes(query)

	var topDist [K]int64
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

	if filled == 0 {
		return 0
	}

	var fraudCount int
	for i := 0; i < filled; i++ {
		if topLabel[i] == LabelFraud {
			fraudCount++
		}
	}
	return float32(fraudCount) / float32(filled)
}

func (d *Dataset) topProbes(query *Vector) [NPROBE]uint32 {
	q := [NumDims]int64{
		int64(query[0]), int64(query[1]), int64(query[2]), int64(query[3]),
		int64(query[4]), int64(query[5]), int64(query[6]), int64(query[7]),
		int64(query[8]), int64(query[9]), int64(query[10]), int64(query[11]),
		int64(query[12]), int64(query[13]),
	}

	var topIdx [NPROBE]uint32
	var topDist [NPROBE]int64
	filled := 0
	worst := int64(math.MaxInt64)

	cents := d.centroids
	for c := 0; c < d.nClusters; c++ {
		// read 14 int16 LE values from cents[c*28 : c*28+28]
		base := unsafe.Pointer(&cents[c*DimBytes])
		var sum int64
		for j := 0; j < NumDims; j++ {
			v := int64(*(*int16)(unsafe.Pointer(uintptr(base) + uintptr(j*2))))
			diff := q[j] - v
			sum += diff * diff
		}

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
	topDist *[K]int64, topLabel *[K]byte, filled int,
) int {
	q := [NumDims]int64{
		int64(query[0]), int64(query[1]), int64(query[2]), int64(query[3]),
		int64(query[4]), int64(query[5]), int64(query[6]), int64(query[7]),
		int64(query[8]), int64(query[9]), int64(query[10]), int64(query[11]),
		int64(query[12]), int64(query[13]),
	}

	for i := start; i < end; i++ {
		off := i * EntrySize
		base := unsafe.Pointer(&body[off])
		var sum int64
		for j := 0; j < NumDims; j++ {
			v := int64(*(*int16)(unsafe.Pointer(uintptr(base) + uintptr(j*2))))
			diff := q[j] - v
			sum += diff * diff
		}
		label := body[off+DimBytes]

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

func siftDown(dist *[K]int64, label *[K]byte, i int) {
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

func siftDownProbes(dist *[NPROBE]int64, idx *[NPROBE]uint32, i int) {
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
