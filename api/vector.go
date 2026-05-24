package main

import "time"

type Vector [14]int8

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

// Quantizes a float in [-1, 1] to int8 in [-127, 127].
// -1 is used as a sentinel for missing last_transaction data.
func quantize(v float32) int8 {
	if v <= -1 {
		return -127
	}
	if v < 0 {
		return 0
	}
	if v >= 1 {
		return 127
	}
	return int8(v * 127)
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

func boolDim(b bool) int8 {
	if b {
		return 127
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
	}
	v[DimAmountVsAvg] = quantize(clamp01(amountVsAvg))

	t := tx.RequestedAt.UTC()
	v[DimHourOfDay] = quantize(float32(t.Hour()) / 23.0)
	v[DimDayOfWeek] = quantize(float32(goWeekdayToSpec(t.Weekday())) / 6.0)

	if r.LastTransaction == nil {
		v[DimMinutesSinceLastTx] = -127
		v[DimKmFromLastTx] = -127
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
