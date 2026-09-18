# Da Pergunta a Evidencia

## Resumo em 30 segundos

Este projeto investiga como diferentes estrategias de Information Retrieval encontram artigos cientificos relevantes para uma pergunta. Comparamos uma busca lexical (BM25), uma busca semantica zero-shot (Dense), uma combinacao dos dois sinais (Hybrid) e um CrossEncoder que reordena os melhores candidatos (Hybrid + Reranker). A avaliacao usa tres benchmarks cientificos do BEIR, com nDCG@10 como metrica principal. O melhor metodo depende do dominio: Hybrid obteve o maior nDCG@10 em NFCorpus e SciFact, enquanto Hybrid + Reranker foi o melhor em TREC-COVID.

## Fala de abertura

> Quando um pesquisador faz uma pergunta, encontrar documentos que contenham as mesmas palavras nem sempre e suficiente. Artigos relevantes podem empregar sinonimos, termos tecnicos ou formulacoes diferentes. Nosso experimento compara busca lexical, busca semantica, combinacao hibrida e reranqueamento neural para verificar quando cada sinal ajuda na recuperacao de literatura cientifica.

## Pergunta e hipotese

**Pergunta de pesquisa:** em que medida modelos semanticos e hibridos melhoram a identificacao de literatura cientifica relevante em comparacao com metodos lexicais?

**Hipotese:** a combinacao dos sinais lexical e semantico, seguida de reranqueamento, pode produzir rankings mais relevantes, embora o ganho possa variar entre dominios.

## Como a aplicacao funciona

1. O usuario escolhe NFCorpus, SciFact ou TREC-COVID.
2. A mesma pergunta e enviada aos modelos selecionados.
3. BM25 procura correspondencia de termos.
4. Dense compara vetores que representam significado semantico.
5. Hybrid combina scores normalizados de BM25 e Dense com `alpha=0.5`.
6. Hybrid + Reranker reordena os 20 primeiros candidatos usando um CrossEncoder.
7. Para queries oficiais do benchmark, qrels permitem mostrar relevancia e metricas por consulta.
8. Para consultas livres, a aplicacao mostra rankings, mas nao calcula qualidade por nao existir ground truth.

O **Pulso do ranking** e uma visualizacao descritiva. Ele mostra o maior nDCG@10 da query, artigos presentes em mais de um ranking e artigos exclusivos de um modelo. Concordancia entre modelos nao significa automaticamente relevancia.

## Os quatro modelos

### BM25

Baseline lexical. Valoriza documentos que contem os termos da consulta, considerando frequencia do termo, raridade na colecao e tamanho do documento. E forte quando a terminologia da pergunta coincide com a dos artigos.

### Dense Retrieval

Usa `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` sem fine-tuning. Consulta e documentos sao convertidos em embeddings normalizados; a busca usa similaridade cosseno por meio de produto interno no FAISS. Pode aproximar textos semanticamente relacionados mesmo sem compartilharem todas as palavras.

### Hybrid

Combina BM25 e Dense depois de normalizar os scores por query. A configuracao final usa fusao ponderada com `alpha=0.5`, escolhida somente no conjunto dev do NFCorpus. Essa configuracao foi congelada antes das avaliacoes finais.

### Hybrid + Reranker

Parte dos candidatos do Hybrid e usa `cross-encoder/ms-marco-MiniLM-L-2-v2` para pontuar conjuntamente cada par pergunta-documento. O CrossEncoder apenas reordena os 20 candidatos; ele nao recupera novos documentos.

## Explicacao das metricas

### nDCG@10, metrica principal

Mede a qualidade das dez primeiras posicoes. Recompensa documentos mais relevantes colocados mais perto do topo e respeita relevancia graduada. O valor varia de 0 a 1; quanto maior, melhor. E a principal metrica porque o usuario costuma examinar primeiro o inicio do ranking.

### Precision@10, ou P@10

E a proporcao dos dez primeiros resultados que foi julgada relevante. `P@10 = 0.70` significa que sete dos dez primeiros documentos sao relevantes. Nesta metrica, qualquer label maior que zero e tratado como relevante.

### Recall@100

E a fracao de todos os documentos relevantes conhecidos que aparece entre os cem primeiros resultados. Recall alto indica boa cobertura de candidatos. Ele nao informa se esses documentos ficaram nas primeiras posicoes.

### MRR

Mean Reciprocal Rank considera a posicao do primeiro documento relevante. Se o primeiro relevante esta na posicao 1, o valor da query e 1; na posicao 2, e 0.5; na posicao 10, e 0.1. A media e calculada entre as queries. MRR responde: quao cedo aparece o primeiro resultado util?

### MAP

Mean Average Precision considera todas as posicoes em que documentos relevantes aparecem e premia rankings que os concentram mais cedo. Primeiro se calcula Average Precision para cada query; depois se tira a media. Neste projeto, o calculo usa o ranking recuperado disponivel, limitado pelo Top K gerado.

## Resultados principais

| Modelo | NFCorpus | SciFact | TREC-COVID |
| --- | ---: | ---: | ---: |
| BM25 | 0.310981 | 0.664390 | 0.567813 |
| Dense | 0.299714 | 0.544566 | 0.568146 |
| Hybrid | **0.335090** | **0.669844** | 0.657468 |
| Hybrid + Reranker | 0.330537 | 0.644935 | **0.708294** |

Valores de nDCG@10. O maior resultado de cada dataset esta em destaque.

### Leitura correta

- No NFCorpus, Dense aumentou Recall@100, mas ficou abaixo de BM25 em nDCG@10, MRR e MAP. Isso indica maior cobertura com ordenacao inicial menos eficaz.
- Hybrid apresentou o melhor nDCG@10 no NFCorpus, com ganho relativo de 7,75% sobre BM25.
- No SciFact, BM25 foi muito competitivo. Hybrid teve o maior nDCG@10, mas o ganho de 0,82% sobre BM25 foi pequeno.
- No TREC-COVID, Hybrid + Reranker atingiu nDCG@10 de 0.708294, ganho relativo de 24,74% sobre BM25.
- O reranker nao melhorou todos os datasets. Ele ajudou claramente no TREC-COVID, mas reduziu o nDCG@10 em NFCorpus e SciFact em relacao ao Hybrid.

## Significancia estatistica

Foi usado paired bootstrap com 10.000 amostras e seed 42 sobre nDCG@10 por query.

- NFCorpus, Hybrid vs BM25: diferenca media `+0.0241`, IC95% `[0.0105, 0.0382]`, `p=0.0012`.
- SciFact, Hybrid vs BM25: diferenca media `+0.0055`, IC95% `[-0.0174, 0.0286]`, `p=0.6602`. O intervalo inclui zero; nao ha evidencia estatistica de diferenca nessa comparacao.
- TREC-COVID, Hybrid + Reranker vs BM25: diferenca media `+0.1405`, IC95% `[0.0954, 0.1875]`, `p<0.0002` na resolucao de 10.000 amostras.
- TREC-COVID, Hybrid + Reranker vs Hybrid: diferenca media `+0.0508`, IC95% `[0.0179, 0.0850]`, `p=0.0032`.

Nao foi aplicada correcao para multiplas comparacoes. Portanto, os p-values devem ser interpretados como exploratorios e em conjunto com magnitude do efeito e intervalo de confianca.

## Protocolo e cuidados metodologicos

- O `alpha=0.5` foi escolhido exclusivamente no NFCorpus dev.
- NFCorpus test foi usado apenas para avaliacao final das configuracoes congeladas.
- SciFact e TREC-COVID foram usados como avaliacao externa zero-shot.
- Qrels nao entram como feature e nao alteram rankings; servem somente para avaliacao.
- O modelo Dense nao recebeu fine-tuning.
- Documentos nao julgados nao devem ser interpretados automaticamente como irrelevantes.
- A demo de consulta livre nao permite afirmar que um ranking e melhor, pois nao ha ground truth.

## Roteiro sugerido de 8 minutos

1. **Problema, 1 minuto:** volume de literatura e diferenca entre palavras iguais e significado semelhante.
2. **Metodos, 1 minuto:** BM25, Dense, Hybrid e CrossEncoder.
3. **Protocolo, 1 minuto:** tres datasets, tuning apenas no NFCorpus dev e avaliacao zero-shot externa.
4. **Metricas, 1 minuto:** destaque nDCG@10; contraste P@10, Recall@100, MRR e MAP.
5. **Demo, 2 minutos:** selecione uma query oficial, abra Comparar e use o Pulso do ranking para mostrar concordancias e mudancas de posicao.
6. **Resultados, 1 minuto:** Hybrid lidera NFCorpus e SciFact; Hybrid + Reranker lidera TREC-COVID.
7. **Conclusao e limites, 1 minuto:** complementaridade existe, mas reranking e ganho semantico dependem do dominio.

## Frase de encerramento

> Os resultados nao mostram um vencedor universal. Eles mostram que sinais lexicais e semanticos sao complementares, que a combinacao hibrida foi mais robusta nos tres benchmarks e que o beneficio do reranqueamento depende do dominio e deve ser validado estatisticamente.
