# Fraud Detection — Rinha de Backend 2026

Submissão para o desafio [Rinha de Backend 2026 — Fraud Detection via Vector Search](https://github.com/ronicarvalho/rinha-de-backend-2026).

## Stack

- **Go 1.23** + [fasthttp](https://github.com/valyala/fasthttp) (servidor) + [sonic](https://github.com/bytedance/sonic) (JSON).
- **nginx** como load balancer round-robin na porta 9999.
- 2 instâncias da API + nginx, somando 1.0 CPU e 350 MB.

## Como funciona a detecção

A API expõe `POST /fraud-score`. Para cada transação:

1. **Normaliza** o payload em um vetor de 14 dimensões (ordem fixa pela spec) usando as constantes de `data/normalization.json` e o lookup de `data/mcc_risk.json`.
2. **Quantiza** cada dimensão de `float32 [-1, 1]` para `int16 [-10000, 10000]` (o `-10000` continua significando "sem `last_transaction`"). 10 000 níveis por dimensão evitam o ruído de quantização que int8 introduzia em casos limítrofes.
3. **Busca os 5 vizinhos mais próximos** no dataset de referência (3 milhões de vetores) usando distância euclidiana ao quadrado, com a estratégia rápido/lento descrita abaixo.
4. **Vota**: `fraud_score = fraud_count / 5`; `approved = fraud_score < 0.6` (literal da spec).

### Estratégia rápido/lento (IVF + bbox repair)

Inspirada no projeto C++ líder. Para cada query:

- **Caminho rápido**: varre apenas os `NPROBE=4` clusters mais próximos da query e calcula o top-5. Se a votação é **unânime** (0 ou 5 frauds), retorna direto — caso da maioria das requests.
- **Caminho lento (`repair`)**: quando a votação é ambígua (1–4 frauds), o algoritmo computa, para cada cluster restante, um **lower bound exato** da menor distância possível entre a query e qualquer ponto do cluster (via bounding box `min/max` por dimensão). Clusters cujo lb já é maior que o pior do top-5 são descartados; os candidatos restantes são varridos em ordem crescente de lb, encolhendo o top-5 e podando agressivamente o resto.

Resultado: o KNN é praticamente exato quando importa (regiões ambíguas) e quase grátis quando não importa (votações unânimes).

## Arquitetura do índice — IVF (Inverted File)

Brute-force em 3 M vetores ≈ 56 ms por query — longe demais do teto de pontuação. A solução é um índice IVF construído em build-time:

- **Build (`api/cmd/preprocess`)**: roda **k-means mini-batch** (`K=2048` centroides, 30 iterações, batches de 8 192) sobre o dataset quantizado, atribui cada entrada ao centroide mais próximo (paralelizado por goroutines) e grava um binário `references.bin` agrupado por cluster.
- **Runtime**: a API faz `mmap` desse arquivo (read-only, `MAP_SHARED`) e, a cada query, varre só os **`NPROBE=16` clusters mais próximos** do vetor de busca (~23 k entradas em vez de 3 M).

Resultado: ~109× de speedup no KNN puro e p99 ~2 ms ponta-a-ponta (vs 94 ms com brute-force).

## Layout binário do dataset

Tudo cabe em ~87 MB, mmap'ado e compartilhável entre as duas instâncias via page cache do kernel:

```
Header (32 bytes):  magic "IVF3" | version | n_entries | n_clusters | n_dims
Centroides:         n_clusters × 28 bytes (14 × int16 LE)
BBoxMin:            n_clusters × 28 bytes (14 × int16 LE)  ← min por dim/cluster
BBoxMax:            n_clusters × 28 bytes (14 × int16 LE)  ← max por dim/cluster
Offsets:            (n_clusters + 1) × uint32  (início de cada cluster)
Entries:            n_entries × 30 bytes (14×int16 + 1 byte label + 1 byte pad)
```

Adicional sobre o IVF1/IVF2 anterior: as duas seções de bbox somam ~112 KB para 2048 clusters — trivial.

## Estrutura

```
api/
  main.go              # boot: carrega config, mmap dataset, sobe fasthttp
  handler.go           # rotas /ready e /fraud-score, structs do payload
  vector.go            # 14 dimensões + quantização int16 (Scale=10000)
  dataset.go           # mmap + busca IVF em 2 etapas
  config.go            # leitura de mcc_risk.json e normalization.json
  cmd/preprocess/      # k-means + writer do references.bin
  Dockerfile           # multi-stage: builder Go -> preprocess -> distroless
data/
  references.json.gz   # 3M vetores rotulados (oficial do desafio)
  mcc_risk.json
  normalization.json
nginx/nginx.conf       # LB round-robin keepalive na 9999
docker-compose.yml    # 2 APIs + nginx, limites 0.45/0.45/0.10 CPU
image-builder.sh       # docker buildx --platform linux/amd64 --push
```

## Rodando localmente

```bash
./image-builder.sh         # builda e dá push em rpxc/fraud-detector-api:latest
docker compose up          # sobe nginx + 2 APIs
curl http://localhost:9999/ready
curl -X POST http://localhost:9999/fraud-score \
  -H 'content-type: application/json' \
  --data data/example-payloads.json  # use uma das transações de exemplo, existem várias no array
```

O Dockerfile detecta se `data/references.json.gz` existe e roda o preprocessador no build; caso contrário, gera um dataset sintético de 100 k entradas para teste.
