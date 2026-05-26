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
	IVFMagic   = uint32(0x33465649) // "IVF3" — adds per-cluster bbox min/max
	HeaderSize = 32
	NumDims    = 14
	DimBytes   = NumDims * 2  // 28
	EntrySize  = DimBytes + 2 // 30: 14*int16 + 1 label + 1 pad
	LabelFraud = 1
	LabelLegit = 0
	K          = 5
	NPROBE     = 8  // fast-path: more unanimous top-5s hit here, avoiding repair
	RepairCap  = 16 // bounded repair: only scan the 16 clusters with smallest bbox lb
)

type Dataset struct {
	raw       []byte   // entire mmap region
	centroids []byte   // [nClusters * DimBytes]
	bboxMin   []byte   // [nClusters * DimBytes]
	bboxMax   []byte   // [nClusters * DimBytes]
	offsets   []uint32 // [nClusters + 1] start of each cluster (in entries)
	body      []byte   // [n * EntrySize]
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
	bmnStart := centEnd
	bmnEnd := bmnStart + nClusters*DimBytes
	bmxStart := bmnEnd
	bmxEnd := bmxStart + nClusters*DimBytes
	offStart := bmxEnd
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
		bboxMin:   data[bmnStart:bmnEnd],
		bboxMax:   data[bmxStart:bmxEnd],
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
	d.bboxMin = nil
	d.bboxMax = nil
	d.body = nil
	return err
}

func (d *Dataset) Len() int       { return d.n }
func (d *Dataset) NClusters() int { return d.nClusters }

// top5 keeps the K nearest neighbors in a max-heap (topDist[0] = worst kept).
type top5 struct {
	dist   [K]int64
	label  [K]byte
	filled int
}

func (t *top5) maxDist() int64 {
	if t.filled < K {
		return math.MaxInt64
	}
	return t.dist[0]
}

func (t *top5) fraudCount() int {
	var n int
	for i := 0; i < t.filled; i++ {
		if t.label[i] == LabelFraud {
			n++
		}
	}
	return n
}

// FraudScore returns n_fraud_in_top5 / 5.
// Fast path: scan NPROBE=4 nearest clusters. If the 5 nearest neighbors are
// unanimous (all fraud or all legit), trust the verdict. Otherwise repair():
// scan every remaining cluster, pruning by bbox lower bound vs. current top5.
func (d *Dataset) FraudScore(query *Vector) float32 {
	probes := d.topProbes(query)

	var top top5
	for _, c := range probes {
		start := int(d.offsets[c])
		end := int(d.offsets[c+1])
		if start == end {
			continue
		}
		d.scanRange(query, start, end, &top)
	}

	fc := top.fraudCount()
	if top.filled == K && (fc == 0 || fc == K) {
		return float32(fc) / float32(K)
	}

	d.repair(query, probes, &top)
	if top.filled == 0 {
		return 0
	}
	return float32(top.fraudCount()) / float32(top.filled)
}

// topProbes finds the NPROBE nearest centroids to the query.
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

// scanRange scans entries [start, end) against the query, updating top.
func (d *Dataset) scanRange(query *Vector, start, end int, top *top5) {
	body := d.body
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

		if top.filled < K {
			top.dist[top.filled] = sum
			top.label[top.filled] = label
			top.filled++
			if top.filled == K {
				for k := K/2 - 1; k >= 0; k-- {
					siftDown(&top.dist, &top.label, k)
				}
			}
			continue
		}
		if sum < top.dist[0] {
			top.dist[0] = sum
			top.label[0] = label
			siftDown(&top.dist, &top.label, 0)
		}
	}
}

// bboxLowerBound returns the minimum possible squared distance from query to
// any vector in cluster c, using its per-dimension bbox.
func (d *Dataset) bboxLowerBound(c uint32, q *Vector) int64 {
	mnBase := unsafe.Pointer(&d.bboxMin[uintptr(c)*uintptr(DimBytes)])
	mxBase := unsafe.Pointer(&d.bboxMax[uintptr(c)*uintptr(DimBytes)])
	var sum int64
	for j := 0; j < NumDims; j++ {
		qv := int64(q[j])
		mn := int64(*(*int16)(unsafe.Pointer(uintptr(mnBase) + uintptr(j*2))))
		mx := int64(*(*int16)(unsafe.Pointer(uintptr(mxBase) + uintptr(j*2))))
		var gap int64
		switch {
		case qv < mn:
			gap = mn - qv
		case qv > mx:
			gap = qv - mx
		}
		sum += gap * gap
	}
	return sum
}

type bboxCand struct {
	lb int64
	c  uint32
}

// repair: when fast path is ambiguous, run a bounded "almost-exact" pass.
// We keep only the RepairCap clusters with smallest bbox lower bound (via a
// max-heap fixed-size on the stack — zero allocation), sort those ascending,
// and scan them in order. The bbox lb is a mathematically exact lower bound
// on the distance to any vector in that cluster, so the RepairCap clusters
// with smallest lb almost certainly contain the true top-K. Once max_top
// tightens below a candidate's lb, we stop scanning the rest.
func (d *Dataset) repair(query *Vector, skip [NPROBE]uint32, top *top5) {
	var heap [RepairCap]bboxCand
	heapSize := 0
	worst := int64(math.MaxInt64)
	maxTop := top.maxDist()

	for c := uint32(0); c < uint32(d.nClusters); c++ {
		skipped := false
		for i := 0; i < NPROBE; i++ {
			if skip[i] == c {
				skipped = true
				break
			}
		}
		if skipped {
			continue
		}
		if d.offsets[c] == d.offsets[c+1] {
			continue
		}
		lb := d.bboxLowerBound(c, query)
		if lb >= maxTop {
			continue
		}
		if heapSize < RepairCap {
			heap[heapSize] = bboxCand{lb, c}
			heapSize++
			if heapSize == RepairCap {
				for k := RepairCap/2 - 1; k >= 0; k-- {
					siftDownCands(&heap, k, RepairCap)
				}
				worst = heap[0].lb
			}
			continue
		}
		if lb < worst {
			heap[0] = bboxCand{lb, c}
			siftDownCands(&heap, 0, RepairCap)
			worst = heap[0].lb
		}
	}

	// Sort heap[0:heapSize] ascending by lb. Small (≤64) — insertion sort
	// beats quicksort on this size and has zero allocation.
	for i := 1; i < heapSize; i++ {
		v := heap[i]
		j := i - 1
		for j >= 0 && heap[j].lb > v.lb {
			heap[j+1] = heap[j]
			j--
		}
		heap[j+1] = v
	}

	for i := 0; i < heapSize; i++ {
		if heap[i].lb >= top.maxDist() {
			break
		}
		start := int(d.offsets[heap[i].c])
		end := int(d.offsets[heap[i].c+1])
		d.scanRange(query, start, end, top)
	}
}

func siftDownCands(h *[RepairCap]bboxCand, i, n int) {
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		max := l
		if l+1 < n && h[l+1].lb > h[l].lb {
			max = l + 1
		}
		if h[i].lb >= h[max].lb {
			return
		}
		h[i], h[max] = h[max], h[i]
		i = max
	}
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
