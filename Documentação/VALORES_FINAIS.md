# Valores finais (dataset v4) — para colar no relatório e nos slides

Configuração travada: split com validação ampliada (60 imagens / 50 instâncias),
teste = Avenida Paulista (195 imagens / 33 instâncias), holdout geográfico.
Limiar de avaliação: confiança 0,25 / IoU 0,50.

## Volumes (seção 3.3 do relatório)

- Treino: [PREENCHER: N treino] imagens
- Validação: 60 imagens (50 instâncias)
- Teste: 195 imagens (33 instâncias) — Avenida Paulista

## Tabela de comparação de arquiteturas (seção 4.1)

| Modelo  | mAP@0.5 | mAP@0.5:0.95 | Precisão | Revocação |
|---------|---------|--------------|----------|-----------|
| YOLO26n | 0,709   | 0,530        | 0,710    | 0,817     |
| YOLO11n | 0,781   | 0,536        | 0,869    | 0,818     |

Validação do YOLO11n (referência): P 0,961 · R 0,977 · mAP@0.5 0,971 · mAP@0.5:0.95 0,760.

## Discussão honesta da comparação (colar abaixo da tabela)

"Na avaliação final, o YOLO11n apresentou desempenho superior ao YOLO26n no conjunto
de teste: precisão de 0,869 contra 0,710 e mAP@0.5 de 0,781 contra 0,709, com
revocação praticamente igual (0,818 contra 0,817) e mAP@0.5:0.95 equivalente (0,536
contra 0,530). Cabe registrar que, em versões anteriores do conjunto de dados, a
ordem entre os dois modelos se invertia, com o YOLO26n à frente. Essa alternância é
coerente com a alta variância de um conjunto de teste com apenas 33 instâncias, em
que diferenças de poucos pontos de mAP não são estatisticamente significativas.
Mantive o YOLO26n como modelo principal por ter sido a escolha de projeto a priori e
por suas melhorias voltadas a objetos pequenos, mas, para este problema e com este
volume de dados, os dois modelos nano devem ser considerados equivalentes."

## Observação sobre variância (manter na seção 4.1)

"O conjunto de teste foi varrido por inteiro, então a maioria dos tiles é de fundo,
sem o alvo. Isso, somado ao número limitado de instâncias, aumenta a variância das
métricas, que devem ser lidas com cautela."

## Conclusão (seção 6) — preenchida

"No conjunto de teste (Avenida Paulista, bairro inédito), o modelo principal
(YOLO26n) encontrou cerca de 82% dos helipontos (revocação 0,817), com precisão de
0,710 e mAP@0.5 de 0,709. A arquitetura de comparação (YOLO11n) atingiu mAP@0.5 de
0,781 e precisão de 0,869 no mesmo teste, ficando à frente nesta versão do dataset —
diferença dentro da margem de variância de um teste pequeno. A divisão por bairro
(holdout geográfico) forneceu uma medida honesta de generalização.

A principal lição confirmada foi a de que o resultado é limitado pelos dados, e não
pela arquitetura: trocar de modelo (nano entre YOLO26 e YOLO11) ou ampliar
marginalmente a base não moveu o resultado de forma significativa. Ganhos reais
dependeriam de um aumento substancial do volume de dados anotados."

## Slides — atualização

### Slide 10 (Resultados)
Tabela:

| Modelo | mAP@0.5 | mAP@0.5:0.95 | Precisão | Revocação |
|--------|---------|--------------|----------|-----------|
| YOLO26n | 0,709 | 0,530 | 0,710 | 0,817 |
| YOLO11n | 0,781 | 0,536 | 0,869 | 0,818 |

Fala (slide 10): "No teste, o YOLO11n ficou ligeiramente à frente do YOLO26n,
principalmente em precisão. Mas a diferença está dentro do ruído de um teste com 33
instâncias — em versões anteriores a ordem se invertia. Na prática, os dois modelos
nano são equivalentes para este problema."

### Slide 14 (Conclusão)
- Resultado no bairro inédito: mAP@0.5 ~0,71–0,78 (modelos equivalentes)
- Recall ~0,82: encontra a maioria dos helipontos
- Lição: o teto é dado pelos dados, não pela arquitetura
- IA generativa: apoio em código/texto; decisões e análise de autoria própria

## Ainda pendente (você me manda e eu preencho)

- Volume de treino da v4 (N de imagens de treino nos logs).
- mAP@0.5 e mAP@0.5:0.95 de validação das rodadas 1 (30 ep) e 2 (50 ep) do YOLO26n,
  para a tabela da seção 3.4.
- Confirmar a matriz de confusão da v4 (TP / FP / FN) para a Figura 7.
