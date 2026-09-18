# Scientific IR Benchmark

Projeto em Python para comparar recuperacao lexical, semantica e hibrida em benchmarks cientificos do BEIR. 

Download do NFCorpus, loader, BM25, metricas e execucao ponta a ponta.

## Objetivo Cientifico

Investigar em que medida modelos de recuperacao semantica e hibrida melhoram a identificacao de literatura cientifica relevante em comparacao com metodos lexicais tradicionais.

## Fase 1

Componentes implementados:

- download automatico do NFCorpus no formato BEIR
- carregamento de corpus, queries e qrels
- baseline BM25 sobre `title + text`
- metricas: nDCG@10, Recall@100, Precision@10, MRR e MAP
- ranking por query e metricas agregadas salvos em `outputs/`

## Instalacao

Use Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Baixar NFCorpus

```bash
python scripts/01_download_datasets.py --dataset nfcorpus
```

## Executar BM25

```bash
python scripts/02_run_bm25.py --dataset nfcorpus --split test
```

Saidas:

- `outputs/rankings/nfcorpus_bm25_test.json`
- `outputs/metrics/nfcorpus_bm25_test.json`
- `outputs/metrics/nfcorpus_bm25_test.csv`
- `outputs/metrics/nfcorpus_bm25_test_per_query.csv`

## Demo Streamlit

A aplicacao interativa em `app.py` reutiliza os datasets, rankings, embeddings e metricas ja produzidos pelo benchmark. Ela permite comparar BM25, Dense, Hybrid e Hybrid + Reranker em NFCorpus, SciFact e TREC-COVID sem alterar os resultados experimentais salvos.

```bash
streamlit run app.py
```

Comportamento da demo:

- se a consulta digitada for um `query_id` ou o texto exato de uma query do benchmark, os rankings salvos em `outputs/rankings/` sao usados e os qrels do split `test` permitem mostrar nDCG@10, Precision@10 e Recall@100 daquela query;
- se a consulta for livre, a busca e executada ao vivo com os mesmos componentes do benchmark, mas sem metricas de relevancia porque nao ha ground truth;
- o Hybrid usa a configuracao congelada `weighted`, `alpha=0.5`;
- o Reranker usa `candidate_pool=20`, igual ao benchmark final;
- os embeddings de corpus sao lidos de `outputs/embeddings/` quando o cache esta disponivel.

## Cuidados Contra Data Leakage

O BM25 não usa qrels para treinar nem ajustar hiperparametros. Os qrels sao usados somente para avaliação do ranking produzido.

Para fases futuras:

- NFCorpus `train` sera usado apenas para fine-tuning.
- NFCorpus `dev` sera usado para escolha de hiperparametros, como alpha do hibrido.
- NFCorpus `test` sera usado somente para avaliacao final.
- SciFact e TREC-COVID nao serao usados para treinamento.

