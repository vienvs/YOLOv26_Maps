# Trecho para o relatório — Efeito do tamanho da validação (colar na Seção 4)

## 4.x Efeito do tamanho do conjunto de validação

Para entender o impacto da divisão treino/validação, avaliei o modelo sob duas
proporções de validação, mantendo a Avenida Paulista sempre como teste inédito e
todos os demais hiperparâmetros idênticos.

| Validação | mAP@0.5 | mAP@0.5:0.95 | Precisão | Revocação |
|-----------|---------|--------------|----------|-----------|
| 10%       | 0,754   | 0,512        | 0,807    | 0,697     |
| 15%       | 0,709   | 0,530        | 0,710    | 0,817     |

Efeitos observados:

- Ampliar a validação de 10% para 15% **reduz a quantidade de dados de treino** e
  altera a seleção do melhor checkpoint, deslocando o ponto de operação do modelo.
- O modelo com 15% de validação ficou **mais sensível** (revocação subiu de 0,697
  para 0,817), encontrando mais helipontos, mas com **mais falsos alarmes**
  (precisão caiu de 0,807 para 0,710).
- As métricas independentes de limiar quase não mudaram: o mAP@0.5 caiu levemente
  (0,754 para 0,709) e o mAP@0.5:0.95 subiu levemente (0,512 para 0,530). Isso
  indica que a **qualidade geral do detector é praticamente a mesma**; o que mudou
  foi o equilíbrio entre precisão e revocação.
- O F1 (média harmônica de precisão e revocação) ficou estável: cerca de 0,75 com
  10% e 0,76 com 15%.

Conclusão do experimento: as diferenças entre as duas proporções estão **dentro da
margem de variância** esperada para um conjunto de teste pequeno (poucas dezenas de
instâncias). A escolha entre as duas configurações é, na prática, a escolha do ponto
de operação desejado: priorizar precisão (10%) ou revocação (15%). O limiar de
confiança no app permite ajustar esse equilíbrio sem retreinar.
