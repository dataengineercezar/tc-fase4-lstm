# Tech Challenge Fase 4 - LSTM para previsao de fechamento de acoes

Este projeto entrega a pipeline completa pedida no desafio:

- coleta de dados historicos via Yahoo Finance;
- pre-processamento temporal sem vazamento de dados;
- modelo LSTM em PyTorch para prever o preco de fechamento;
- avaliacao com MAE, RMSE e MAPE;
- exportacao do modelo, scaler e metadados;
- API REST com FastAPI;
- endpoint `/metrics` com metricas Prometheus;
- Docker e Docker Compose para deploy.

## Decisao tecnica

O modelo usa somente a serie `Close` por padrao. Isso evita depender de variaveis futuras como volume, maxima e minima para prever dias adiante. A API aceita uma janela historica de fechamentos e retorna previsoes autoregressivas para os proximos dias uteis.

O simbolo padrao e `AAPL`, mas voce pode usar qualquer ticker suportado pelo Yahoo Finance, por exemplo `PETR4.SA`, `VALE3.SA`, `MSFT` ou `DIS`. Treine um modelo por ativo; a API recusa previsoes Yahoo para um ticker diferente daquele usado no treino.

## Estrutura

```text
src/stock_lstm/
  api.py          # API FastAPI
  config.py       # configuracoes do projeto
  data.py         # coleta e normalizacao de dados
  features.py     # criacao de janelas supervisionadas
  metrics.py      # metricas de regressao
  model.py        # arquitetura LSTM
  predict.py      # carregamento de artefatos e inferencia
  train.py        # treino, avaliacao e salvamento
tests/
  test_features.py
  test_metrics.py
```

## Como rodar localmente

Recomendado: Python 3.10, 3.11 ou 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Treine o modelo:

```powershell
python -m stock_lstm.train --symbol AAPL --start 2018-01-01 --lookback 60 --epochs 50 --patience 8 --batch-size 64
```

Ao final, os artefatos serao salvos em `artifacts/`:

- `model.pt`
- `target_scaler.joblib`
- `metadata.json`
- `metrics.json`

Este repositorio ja inclui artefatos treinados para `AAPL` (ultimo treino: 2026-05-20, early stopping no 14° epoch), permitindo subir a API diretamente depois de instalar as dependencias ou construir a imagem Docker.

Suba a API:

```powershell
uvicorn stock_lstm.api:app --reload --host 0.0.0.0 --port 8000
```

Abra:

- http://localhost:8000/docs
- http://localhost:8000/health
- http://localhost:8000/metrics

## Exemplo de chamada da API

Use o endpoint de conveniencia para baixar historico recente pelo Yahoo Finance e prever os proximos dias:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/predict/yfinance `
  -ContentType "application/json" `
  -Body '{"symbol":"AAPL","lookback_days":180,"horizon":5}'
```

Ou envie diretamente os fechamentos historicos:

```json
{
  "horizon": 3,
  "prices": [
    {"date": "2026-05-19", "close": 189.12},
    {"date": "2026-05-20", "close": 190.33}
  ]
}
```

Para esse endpoint, envie pelo menos `lookback` observacoes historicas, por padrao 60.

## Docker

Treinar usando Docker Compose:

```powershell
docker compose run --rm trainer
```

Subir API:

```powershell
docker compose up api
```

## Monitoramento

A API publica metricas em `/metrics`, incluindo:

- total de requisicoes por endpoint e status;
- latencia por endpoint;
- status de carregamento do modelo;
- horizonte usado nas previsoes.

Em producao, esse endpoint pode ser coletado por Prometheus e visualizado no Grafana.

## Roteiro sugerido para o video

1. Explique o problema: prever fechamento de uma acao com LSTM.
2. Mostre a coleta com `yfinance` e a separacao temporal treino/validacao/teste.
3. Mostre a arquitetura LSTM e as metricas finais em `artifacts/metrics.json`.
4. Rode a API e abra `/docs`.
5. Execute uma previsao via `/predict/yfinance`.
6. Mostre `/metrics` e explique o monitoramento.
7. Mostre Docker Compose como caminho de deploy.

## Checklist de qualidade

- Split temporal, sem embaralhar validacao/teste.
- Scaler ajustado apenas no periodo de treino.
- Artefatos versionaveis fora do codigo-fonte.
- API com validacao de entrada.
- Metricas tecnicas e operacionais.
- Docker para reproduzir o deploy.
