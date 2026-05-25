package main

import "time"

const Scale = 10000

type Vector [14]int16

const (
	DimAmount             = 0
	DimInstallments       = 1
	DimAmountVsAvg        = 2
	DimHourOfDay          = 3
	DimDayOfWeek          = 4
	DimMinutesSinceLastTx = 5
	DimKmFromLastTx       = 6
	DimKmFromHome         = 7
	DimTxCount24h         = 8
	DimIsOnline           = 9
	DimCardPresent        = 10
	DimUnknownMerchant    = 11
	DimMccRisk            = 12
	DimMerchantAvgAmount  = 13
)

// quantize maps a clamped float in [0, 1] to int16 in [0, Scale],
// using round-half-away-from-zero (matches the winning .NET impl).
func quantize(v float32) int16 {
	x := v * Scale
	if x >= 0 {
		return int16(x + 0.5)
	}
	return int16(x - 0.5)
}

// quantizeReference handles the sentinel: float -1 (missing last_transaction)
// maps to -Scale; otherwise clamp to [0,1] then quantize.
func quantizeReference(v float32) int16 {
	if v <= -0.9999 {
		return -Scale
	}
	return quantize(clamp01(v))
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func boolDim(b bool) int16 {
	if b {
		return Scale
	}
	return 0
}

// goWeekdayToSpec converts Go's weekday (Sun=0..Sat=6) to spec (Mon=0..Sun=6).
func goWeekdayToSpec(wd time.Weekday) int {
	return (int(wd) + 6) % 7
}

func normalize(r *Request, cfg *Config) Vector {
	var v Vector
	tx := r.Transaction

	v[DimAmount] = quantize(clamp01(tx.Amount / cfg.MaxAmount))
	v[DimInstallments] = quantize(clamp01(float32(tx.Installments) / cfg.MaxInstallments))

	var amountVsAvg float32
	if r.Customer.AvgAmount > 0 {
		amountVsAvg = (tx.Amount / r.Customer.AvgAmount) / cfg.AmountVsAvgRatio
	} else {
		amountVsAvg = 1.0
	}
	v[DimAmountVsAvg] = quantize(clamp01(amountVsAvg))

	t := tx.RequestedAt.UTC()
	v[DimHourOfDay] = quantize(float32(t.Hour()) / 23.0)
	v[DimDayOfWeek] = quantize(float32(goWeekdayToSpec(t.Weekday())) / 6.0)

	if r.LastTransaction == nil {
		v[DimMinutesSinceLastTx] = -Scale
		v[DimKmFromLastTx] = -Scale
	} else {
		mins := float32(tx.RequestedAt.Sub(r.LastTransaction.Timestamp).Minutes())
		v[DimMinutesSinceLastTx] = quantize(clamp01(mins / cfg.MaxMinutes))
		v[DimKmFromLastTx] = quantize(clamp01(r.LastTransaction.KmFromCurrent / cfg.MaxKm))
	}

	v[DimKmFromHome] = quantize(clamp01(r.Terminal.KmFromHome / cfg.MaxKm))
	v[DimTxCount24h] = quantize(clamp01(float32(r.Customer.TxCount24h) / cfg.MaxTxCount24h))
	v[DimIsOnline] = boolDim(r.Terminal.IsOnline)
	v[DimCardPresent] = boolDim(r.Terminal.CardPresent)
	v[DimUnknownMerchant] = boolDim(isUnknownMerchant(r.Merchant.Id, r.Customer.KnownMerchants))

	risk, ok := cfg.MccRisk[r.Merchant.Mcc]
	if !ok {
		risk = 0.5
	}
	v[DimMccRisk] = quantize(risk)

	v[DimMerchantAvgAmount] = quantize(clamp01(r.Merchant.AvgAmount / cfg.MaxMerchantAvgAmount))

	return v
}

func isUnknownMerchant(id string, known []string) bool {
	for _, k := range known {
		if k == id {
			return false
		}
	}
	return true
}
