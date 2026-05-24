package main

import (
	"fmt"
	"os"

	"github.com/bytedance/sonic"
)

type Config struct {
	MaxAmount            float32
	MaxInstallments      float32
	AmountVsAvgRatio     float32
	MaxMinutes           float32
	MaxKm                float32
	MaxTxCount24h        float32
	MaxMerchantAvgAmount float32
	MccRisk              map[string]float32
}

type normalizationFile struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKm                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

func loadConfig(normPath, mccPath string) (*Config, error) {
	nb, err := os.ReadFile(normPath)
	if err != nil {
		return nil, fmt.Errorf("read normalization: %w", err)
	}
	var n normalizationFile
	if err := sonic.Unmarshal(nb, &n); err != nil {
		return nil, fmt.Errorf("parse normalization: %w", err)
	}

	mb, err := os.ReadFile(mccPath)
	if err != nil {
		return nil, fmt.Errorf("read mcc_risk: %w", err)
	}
	var m map[string]float32
	if err := sonic.Unmarshal(mb, &m); err != nil {
		return nil, fmt.Errorf("parse mcc_risk: %w", err)
	}

	return &Config{
		MaxAmount:            n.MaxAmount,
		MaxInstallments:      n.MaxInstallments,
		AmountVsAvgRatio:     n.AmountVsAvgRatio,
		MaxMinutes:           n.MaxMinutes,
		MaxKm:                n.MaxKm,
		MaxTxCount24h:        n.MaxTxCount24h,
		MaxMerchantAvgAmount: n.MaxMerchantAvgAmount,
		MccRisk:              m,
	}, nil
}
