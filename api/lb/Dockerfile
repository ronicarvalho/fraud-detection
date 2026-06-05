FROM golang:1.22-alpine AS build
WORKDIR /app
# Context is project root, so we need to point to api/
COPY api/go.mod api/go.sum ./
RUN go mod download
COPY api/ ./
RUN go build -o /lb ./lb/main.go

FROM alpine:latest
WORKDIR /app
COPY --from=build /lb /app/lb
ENV LB_ADDR=0.0.0.0:9999 \
    LB_UPSTREAMS=/sockets/api1.sock,/sockets/api2.sock
ENTRYPOINT ["/app/lb"]
