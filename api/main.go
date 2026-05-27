package main

import (
	"log"
	"os"
	"time"

	"github.com/valyala/fasthttp"
)

var (
	ds  *Dataset
	cfg *Config
)

func main() {
	dataDir := envOrDefault("DATA_DIR", "/app/data")

	var err error
	cfg, err = loadConfig(dataDir+"/normalization.json", dataDir+"/mcc_risk.json")
	if err != nil {
		log.Fatalf("config load: %v", err)
	}
	log.Printf("config loaded (%d mcc entries)", len(cfg.MccRisk))

	ds, err = LoadDataset(dataDir + "/references.bin")
	if err != nil {
		log.Fatalf("dataset load: %v", err)
	}
	defer ds.Close()
	log.Printf("dataset loaded (%d entries)", ds.Len())

	instanceID := envOrDefault("INSTANCE_ID", "?")
	log.Printf("api instance %s listening on :8080", instanceID)

	server := &fasthttp.Server{
		Handler:            handler,
		Name:               "fraud-detector-api",
		MaxConnsPerIP:      4096,
		MaxRequestBodySize: 64 * 1024,
		ReadTimeout:        5 * time.Second,
		WriteTimeout:       5 * time.Second,
		TCPKeepalive:       true,
	}

	if err := server.ListenAndServe(":8080"); err != nil {
		log.Fatalf("server: %v", err)
	}
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
