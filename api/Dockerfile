FROM golang:1.23-alpine AS builder
WORKDIR /src
COPY api/go.mod api/go.sum ./
RUN go mod download
COPY api/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -trimpath -o /out/api ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -trimpath -o /out/preprocess ./cmd/preprocess

FROM alpine:3.19 AS dataset
WORKDIR /work
COPY --from=builder /out/preprocess /usr/local/bin/preprocess
COPY data/ /input/
RUN if [ -f /input/references.json.gz ]; then \
      preprocess /input/references.json.gz /work/references.bin; \
    else \
      echo "WARN: data/references.json.gz missing, generating 100k synthetic entries"; \
      preprocess -synth 100000 /work/references.bin; \
    fi

FROM gcr.io/distroless/static:nonroot
WORKDIR /app
COPY --from=builder /out/api /app/api
COPY --from=dataset /work/references.bin /app/data/references.bin
COPY data/mcc_risk.json data/normalization.json /app/data/
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/app/api"]
