# Fraud Detection — Rinha de Backend 2026

Submissão para o desafio [Rinha de Backend 2026 — Fraud Detection via Vector Search](https://github.com/ronicarvalho/rinha-de-backend-2026).

## Stack

- **Go 1.23** + [fasthttp](https://github.com/valyala/fasthttp) (servidor) + [sonic](https://github.com/bytedance/sonic) (JSON).
- **Custom Load Balancer (Go)** usando **FD Handoff (SCM_RIGHTS)** via Unix Sockets.
- 2 instâncias da API + LB customizado, somando 1.0 CPU e 350 MB.

## Arquitetura de Performance: FD Handoff

Para alcançar a menor latência possível (p99 < 1ms), substituímos o Nginx por um Load Balancer customizado que utiliza a técnica de **File Descriptor Handoff**:

1. **Payload-blind**: O LB não lê nem faz parse do corpo HTTP. Ele apenas aceita a conexão TCP.
2. **SCM_RIGHTS**: Assim que aceita uma conexão, o LB escolhe uma instância da API (Round-Robin) e transfere o File Descriptor (FD) do socket do cliente diretamente para a API através de um Unix Control Socket usando a syscall `sendmsg` com `SCM_RIGHTS`.
3. **Comunicação Direta**: Após o handoff, a API assume a conexão e fala diretamente com o cliente. O LB sai do caminho, eliminando overhead de proxy reverso, buffering e parsing duplicado.

## Como funciona a detecção

A API expõe `POST /fraud-score`. Para cada transação:

1. **Normaliza** o payload em um vetor de 14 dimensões usando as constantes de `data/normalization.json` e o lookup de `data/mcc_risk.json`.
2. **Quantiza** cada dimensão de `float32 [-1, 1]` para `int16 [-10000, 10000]`.
3. **Busca os 5 vizinhos mais próximos** no dataset de referência (3 milhões de vetores) usando distância euclidiana ao quadrado.
4. **Vota**: `fraud_score = fraud_count / 5`; `approved = fraud_score < 0.6`.

### Estratégia rápido/lento (IVF + bbox repair)

- **Caminho rápido**: Varre os clusters (`NPROBE=16`) mais próximos e calcula o top-5. Se a votação for unânime (0 ou 5 frauds), retorna imediatamente.
- **Caminho lento (`repair`)**: Em casos ambíguos, utiliza os bounding boxes (`min/max`) de cada cluster para calcular um *lower bound* exato de distância e podar clusters que não podem alterar o resultado final.

## Arquitetura do índice — IVF (Inverted File)

- **Build-time (`api/cmd/preprocess`)**: Executa **k-means mini-batch** (`K=2048` centroides) para agrupar os 3M de vetores em clusters.
- **Runtime**: A API mapeia o binário resultante via `mmap` (read-only, `MAP_SHARED`), permitindo que as duas instâncias compartilhem a mesma memória física através do Page Cache do kernel.

## Estrutura

```
api/
  main.go              # Server fasthttp + custom FD Handoff listener
  lb/                  # Load Balancer customizado (FD Handoff implementation)
  handler.go           # Rotas /ready e /fraud-score
  vector.go            # Normalização e quantização int16
  dataset.go           # Busca IVF otimizada
  cmd/preprocess/      # Gerador do índice binário (k-means)
  Dockerfile           # Multi-stage build (API)
data/
  references.json.gz   # Dataset oficial (3M vetores)
  mcc_risk.json
  normalization.json
docker-compose.yml     # Orquestração: 2 APIs + LB customizado
```

## Rodando localmente

```bash
# Build e deploy
docker compose up --build -d

# Testar disponibilidade
curl http://localhost:9999/ready

# Enviar transação para score
curl -X POST http://localhost:9999/fraud-score \
  -H 'content-type: application/json' \
  --data @data/example-payloads.json
```

O Load Balancer escuta na porta **9999** e distribui o tráfego para as instâncias da API via sockets localizados no volume compartilhado `sockets`.
