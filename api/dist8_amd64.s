// func dist8AVX2(q *Vector, blockPtr unsafe.Pointer, out *[BlockSize]float32)
//
// Computes squared L2 distance between q (14 int16) and 8 vectors stored in
// pair-SoA layout (7 pairs × 8 lanes × 2 int16 = 224 bytes), returning 8
// float32 in out.
//
// Plan-9 AVX2 operand order: VPSUBW src2, src1, dst means dst = src1 - src2.
// VPMADDWD src2, src1, dst → dst[lane] = src1[lane].lo*src2[lane].lo + src1[lane].hi*src2[lane].hi.
//
// Pair-SoA layout per block:
//   pair p (p=0..6) at byte offset p*32: 16 int16 = lane0.dp0, lane0.dp1, lane1.dp0, lane1.dp1, ..., lane7.dp0, lane7.dp1
//
// Query is read as 7 broadcast pairs from q[0..27].
//
// Two-accumulator pattern (acc0, acc1) reduces dependency chain latency.

#include "textflag.h"

TEXT ·dist8AVX2(SB), NOSPLIT, $0-24
	MOVQ q+0(FP), AX
	MOVQ blockPtr+8(FP), BX
	MOVQ out+16(FP), CX

	// Pre-broadcast 7 query pairs into Y8..Y14
	VPBROADCASTD  0(AX), Y8
	VPBROADCASTD  4(AX), Y9
	VPBROADCASTD  8(AX), Y10
	VPBROADCASTD 12(AX), Y11
	VPBROADCASTD 16(AX), Y12
	VPBROADCASTD 20(AX), Y13
	VPBROADCASTD 24(AX), Y14

	// Two float32 accumulators
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1

	// Pair 0 → acc0
	VMOVDQU   0(BX), Y2
	VPSUBW    Y2, Y8, Y3
	VPMADDWD  Y3, Y3, Y4
	VCVTDQ2PS Y4, Y5
	VADDPS    Y5, Y0, Y0

	// Pair 1 → acc1
	VMOVDQU   32(BX), Y2
	VPSUBW    Y2, Y9, Y3
	VPMADDWD  Y3, Y3, Y4
	VCVTDQ2PS Y4, Y5
	VADDPS    Y5, Y1, Y1

	// Pair 2 → acc0
	VMOVDQU   64(BX), Y2
	VPSUBW    Y2, Y10, Y3
	VPMADDWD  Y3, Y3, Y4
	VCVTDQ2PS Y4, Y5
	VADDPS    Y5, Y0, Y0

	// Pair 3 → acc1
	VMOVDQU   96(BX), Y2
	VPSUBW    Y2, Y11, Y3
	VPMADDWD  Y3, Y3, Y4
	VCVTDQ2PS Y4, Y5
	VADDPS    Y5, Y1, Y1

	// Pair 4 → acc0
	VMOVDQU   128(BX), Y2
	VPSUBW    Y2, Y12, Y3
	VPMADDWD  Y3, Y3, Y4
	VCVTDQ2PS Y4, Y5
	VADDPS    Y5, Y0, Y0

	// Pair 5 → acc1
	VMOVDQU   160(BX), Y2
	VPSUBW    Y2, Y13, Y3
	VPMADDWD  Y3, Y3, Y4
	VCVTDQ2PS Y4, Y5
	VADDPS    Y5, Y1, Y1

	// Pair 6 → acc0
	VMOVDQU   192(BX), Y2
	VPSUBW    Y2, Y14, Y3
	VPMADDWD  Y3, Y3, Y4
	VCVTDQ2PS Y4, Y5
	VADDPS    Y5, Y0, Y0

	// Combine: acc0 += acc1, store
	VADDPS  Y1, Y0, Y0
	VMOVUPS Y0, (CX)

	VZEROUPPER
	RET
