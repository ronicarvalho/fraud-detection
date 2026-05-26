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
	IVFMagic     = uint32(0x34465649) // "IVF4" — pair-SoA blocked layout
	HeaderSize   = 32
	NumDims      = 14
	NumPairs     = NumDims / 2 // 7
	BlockSize    = 8           // entries per block (matches AVX2 lane count)
	BlockVecSize = NumPairs * BlockSize * 2 * 2 // 224 bytes
	LabelFraud   = 1
	LabelLegit   = 0
	K            = 5
	NPROBE       = 8
	RepairCap    = 16
)

type Dataset struct {
	raw          []byte
	centroids    []byte   // [nClusters * 28]
	bboxMin      []byte   // [nClusters * 28]
	bboxMax      []byte   // [nClusters * 28]
	blockOffsets []uint32 // [nClusters + 1]
	counts       []uint32 // [nClusters] real entry count per cluster
	labels       []byte   // [totalBlocks * 8]
	blocks       []byte   // [totalBlocks * 224]
	n            int
	nClusters    int
	totalBlocks  int
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
	totalBlocks := int(binary.LittleEndian.Uint32(data[24:28]))
	if nDims != NumDims {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("dims=%d, expected %d", nDims, NumDims)
	}

	dimBytes := NumDims * 2
	centStart := HeaderSize
	centEnd := centStart + nClusters*dimBytes
	bmnStart := centEnd
	bmnEnd := bmnStart + nClusters*dimBytes
	bmxStart := bmnEnd
	bmxEnd := bmxStart + nClusters*dimBytes
	offStart := bmxEnd
	offEnd := offStart + (nClusters+1)*4
	cntStart := offEnd
	cntEnd := cntStart + nClusters*4
	lblStart := cntEnd
	lblEnd := lblStart + totalBlocks*BlockSize
	blkStart := lblEnd
	blkEnd := blkStart + totalBlocks*BlockVecSize
	if blkEnd > size {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("file size %d < expected %d", size, blkEnd)
	}

	offsets := make([]uint32, nClusters+1)
	for i := 0; i <= nClusters; i++ {
		offsets[i] = binary.LittleEndian.Uint32(data[offStart+i*4:])
	}
	counts := make([]uint32, nClusters)
	for i := 0; i < nClusters; i++ {
		counts[i] = binary.LittleEndian.Uint32(data[cntStart+i*4:])
	}

	return &Dataset{
		raw:          data,
		centroids:    data[centStart:centEnd],
		bboxMin:      data[bmnStart:bmnEnd],
		bboxMax:      data[bmxStart:bmxEnd],
		blockOffsets: offsets,
		counts:       counts,
		labels:       data[lblStart:lblEnd],
		blocks:       data[blkStart:blkEnd],
		n:            n,
		nClusters:    nClusters,
		totalBlocks:  totalBlocks,
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
	d.labels = nil
	d.blocks = nil
	return err
}

func (d *Dataset) Len() int       { return d.n }
func (d *Dataset) NClusters() int { return d.nClusters }

// top5 keeps the K nearest neighbors in a max-heap (dist[0] = worst kept).
// Distances are stored as float32 to match the SIMD path (which uses
// VCVTDQ2PS / VADDPS for accumulation, same as the C++ leader).
type top5 struct {
	dist   [K]float32
	label  [K]byte
	filled int
}

func (t *top5) maxDist() float32 {
	if t.filled < K {
		return math.MaxFloat32
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

func (t *top5) update(dist float32, label byte) {
	if t.filled < K {
		t.dist[t.filled] = dist
		t.label[t.filled] = label
		t.filled++
		if t.filled == K {
			for k := K/2 - 1; k >= 0; k-- {
				siftDown(&t.dist, &t.label, k)
			}
		}
		return
	}
	if dist < t.dist[0] {
		t.dist[0] = dist
		t.label[0] = label
		siftDown(&t.dist, &t.label, 0)
	}
}

// FraudScore implements the fast/slow path strategy:
//   - Fast: scan NPROBE=8 nearest clusters via pair-SoA blocks.
//   - If voting is unanimous (0 or K frauds), return.
//   - Slow (repair): bbox lower bound prunes; scan the RepairCap clusters with smallest lb.
func (d *Dataset) FraudScore(query *Vector) float32 {
	probes := d.topProbes(query)

	var top top5
	for _, c := range probes {
		d.scanCluster(c, query, &top)
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
		base := unsafe.Pointer(&cents[c*NumDims*2])
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

// scanCluster scans every block of cluster c, updating top with the 8 lane
// distances. Uses dist8Scalar (or AVX2 once Fase 2 is in).
func (d *Dataset) scanCluster(c uint32, query *Vector, top *top5) {
	cnt := d.counts[c]
	if cnt == 0 {
		return
	}
	blkStart := d.blockOffsets[c]
	blkEnd := d.blockOffsets[c+1]

	var dists [BlockSize]float32
	for bi := blkStart; bi < blkEnd; bi++ {
		blockPtr := unsafe.Pointer(&d.blocks[uintptr(bi)*uintptr(BlockVecSize)])
		labelPtr := unsafe.Pointer(&d.labels[uintptr(bi)*uintptr(BlockSize)])

		dist8Scalar(query, blockPtr, &dists)

		posInCluster := (bi - blkStart) * BlockSize
		var nValid uint32
		if posInCluster+BlockSize > cnt {
			nValid = cnt - posInCluster
		} else {
			nValid = BlockSize
		}

		for lane := uint32(0); lane < nValid; lane++ {
			label := *(*byte)(unsafe.Pointer(uintptr(labelPtr) + uintptr(lane)))
			top.update(dists[lane], label)
		}
	}
}

// dist8Scalar computes squared euclidean distance between `q` and 8 vectors
// stored pair-SoA inside `blockPtr` (224 bytes). Acumulação em float32 para
// match bit-a-bit com dist8AVX2 (Fase 2).
func dist8Scalar(q *Vector, blockPtr unsafe.Pointer, out *[BlockSize]float32) {
	var sums [BlockSize]float32
	for p := 0; p < NumPairs; p++ {
		q0 := int32(q[p*2])
		q1 := int32(q[p*2+1])
		pairOff := p * BlockSize * 4 // 32 bytes per pair group
		for lane := 0; lane < BlockSize; lane++ {
			laneOff := pairOff + lane*4
			v0 := int32(*(*int16)(unsafe.Add(blockPtr, laneOff)))
			v1 := int32(*(*int16)(unsafe.Add(blockPtr, laneOff+2)))
			d0 := q0 - v0
			d1 := q1 - v1
			// VPMADDWD semantics: (d0*d0 + d1*d1) in int32, then converted to float32 and added.
			sums[lane] += float32(d0*d0 + d1*d1)
		}
	}
	*out = sums
}

func (d *Dataset) bboxLowerBound(c uint32, q *Vector) float32 {
	mnBase := unsafe.Pointer(&d.bboxMin[uintptr(c)*uintptr(NumDims*2)])
	mxBase := unsafe.Pointer(&d.bboxMax[uintptr(c)*uintptr(NumDims*2)])
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
	return float32(sum)
}

type bboxCand struct {
	lb float32
	c  uint32
}

func (d *Dataset) repair(query *Vector, skip [NPROBE]uint32, top *top5) {
	var heap [RepairCap]bboxCand
	heapSize := 0
	worst := float32(math.MaxFloat32)
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
		if d.counts[c] == 0 {
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
		d.scanCluster(heap[i].c, query, top)
	}
}

func siftDown(dist *[K]float32, label *[K]byte, i int) {
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
