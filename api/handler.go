package main

import (
	"bytes"
	"sync"
	"time"

	"github.com/bytedance/sonic"
	"github.com/valyala/fasthttp"
)

type Transaction struct {
	Amount       float32   `json:"amount"`
	Installments int       `json:"installments"`
	RequestedAt  time.Time `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float32  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	Id        string  `json:"id"`
	Mcc       string  `json:"mcc"`
	AvgAmount float32 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float32 `json:"km_from_home"`
}

type LastTransaction struct {
	Timestamp     time.Time `json:"timestamp"`
	KmFromCurrent float32   `json:"km_from_current"`
}

type Request struct {
	Id              string           `json:"id"`
	Transaction     Transaction      `json:"transaction"`
	Customer        Customer         `json:"customer"`
	Merchant        Merchant         `json:"merchant"`
	Terminal        Terminal         `json:"terminal"`
	LastTransaction *LastTransaction `json:"last_transaction"`
}

type Response struct {
	Approved   bool    `json:"approved"`
	FraudScore float32 `json:"fraud_score"`
}

var requestPool = sync.Pool{
	New: func() any {
		return new(Request)
	},
}

var (
	pathFraudScore = []byte("/fraud-score")
	pathReady      = []byte("/ready")
)

func handler(ctx *fasthttp.RequestCtx) {
	path := ctx.Path()
	if bytes.Equal(path, pathFraudScore) {
		fraudScoreHandler(ctx)
	} else if bytes.Equal(path, pathReady) {
		ctx.SetStatusCode(fasthttp.StatusOK)
	} else {
		ctx.SetStatusCode(fasthttp.StatusNotFound)
	}
}

func fraudScoreHandler(ctx *fasthttp.RequestCtx) {
	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	req := requestPool.Get().(*Request)
	defer func() {
		req.Id = ""
		req.Customer.KnownMerchants = nil
		req.LastTransaction = nil
		requestPool.Put(req)
	}()

	if err := sonic.Unmarshal(ctx.PostBody(), req); err != nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		return
	}

	vec := normalize(req, cfg)
	score := ds.FraudScore(&vec)

	res := Response{
		Approved:   score < 0.6,
		FraudScore: score,
	}

	ctx.SetContentType("application/json")
	ctx.SetStatusCode(fasthttp.StatusOK)
	_ = sonic.ConfigDefault.NewEncoder(ctx).Encode(res)
}
