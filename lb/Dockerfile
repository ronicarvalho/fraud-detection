FROM alpine:3.19 AS builder
RUN apk add --no-cache gcc musl-dev
WORKDIR /src
COPY fd_lb.c .
RUN mkdir -p /out && gcc -O2 -static -s -o /out/fd_lb fd_lb.c

FROM scratch
COPY --from=builder /out/fd_lb /fd_lb
ENTRYPOINT ["/fd_lb"]
