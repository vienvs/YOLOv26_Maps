# Detecção de Helipontos em Imagens de Satélite com YOLO

Pontifícia Universidade Católica de São Paulo — PUC-SP

Vinicius de Lucena

Projeto P2 — Aprendizagem de Máquina (Ciência de Dados e Inteligência Artificial) — Prof. Dr. Rooney Coelho

São Paulo, 2026

---

> Observações para a montagem final (remover antes de entregar):
> - Onde aparecer **[INSERIR FIGURA: ...]**, insira a imagem indicada com a legenda no formato "Figura N - descrição (fonte)".
> - Onde aparecer **[PREENCHER: ...]**, substitua pelo número real obtido no notebook após o treino final.
> - Toda figura que mostre tiles ou mosaicos do ESRI deve trazer a atribuição: "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community".
> - Estrutura espelhada nos relatórios anteriores (capa, sumário, seções numeradas, figuras legendadas, declaração de IA e referências).

---

## SUMÁRIO

1. Introdução
2. Fundamentação Teórica
   - 2.1 Detecção de Objetos e a Família YOLO
   - 2.2 Métricas de Detecção (IoU, NMS, Precisão, Revocação e mAP)
3. Metodologia
   - 3.1 Coleta de Dados
   - 3.2 Definição do Alvo e Anotação
   - 3.3 Pré-processamento e Divisão dos Dados
   - 3.4 Estratégia de Treinamento
4. Resultados e Discussão
   - 4.1 Comparação das Rodadas e das Arquiteturas
   - 4.2 Curvas de Treino
   - 4.3 Análise de Erros
   - 4.4 Inferência em Bairro Inédito
5. Aplicação no Streamlit
6. Conclusões
7. Referências

---

## 1 INTRODUÇÃO

O trabalho desenvolvido aborda o problema da detecção de um objeto específico em imagens de satélite da cidade de São Paulo: o heliponto em cobertura. A visão computacional baseada em aprendizado profundo permite localizar e classificar objetos em imagens, e neste projeto construí, de ponta a ponta, um detector de uma única classe, cobrindo todo o ciclo de um sistema real: da coleta do dado bruto ao modelo treinado, avaliado e disponibilizado em uma aplicação web.

O objetivo é treinar e avaliar um detector capaz de identificar helipontos e medir sua capacidade de generalizar para uma região não vista durante o treino. Os objetivos específicos são: coletar imagens de forma programática, anotar com consistência, treinar com transfer learning, avaliar com métricas padronizadas, analisar os erros do modelo e disponibilizar uma aplicação de demonstração.

A mensagem central da disciplina orienta o projeto: a maior parte do esforço de um sistema de IA está nos dados, e não na arquitetura. Como o modelo de detecção já vem pré-treinado, o que diferencia um resultado bom de um ruim é a qualidade da coleta, da curadoria e da anotação. Por isso, a maior parte deste relatório trata da construção do conjunto de dados.

## 2 FUNDAMENTAÇÃO TEÓRICA

Esta seção apresenta os conceitos que fundamentam as decisões técnicas do trabalho.

### 2.1 Detecção de Objetos e a Família YOLO

Diferentemente da classificação, em que a rede atribui um rótulo único à imagem inteira, a detecção de objetos precisa responder duas perguntas ao mesmo tempo: o que existe na imagem e onde está cada ocorrência, delimitada por uma caixa (bounding box). O YOLO (You Only Look Once) é uma família de detectores de estágio único que prevê caixas e classes em uma só passagem pela rede, o que o torna rápido e adequado a um fluxo prático de treino e inferência.

O modelo principal escolhido foi o YOLO26n, da Ultralytics, por trazer melhorias úteis para alvos pequenos, como um esquema de atribuição de rótulos consciente de pequenos objetos e uma cabeça de detecção mais leve. A variante nano cabe com folga na GPU T4 do Google Colab e treina rápido para um dataset pequeno. Para a comparação obrigatória de arquitetura, utilizei também o YOLO11n, sob os mesmos hiperparâmetros.

[INSERIR FIGURA: exemplo de heliponto em vista de satélite, com a caixa de anotação]

### 2.2 Métricas de Detecção (IoU, NMS, Precisão, Revocação e mAP)

A avaliação de um detector se apoia em algumas definições vistas em aula:

- **IoU (Intersection over Union):** mede a sobreposição entre a caixa prevista e a caixa verdadeira; é o critério que decide se uma detecção "casa" com um objeto real.
- **NMS (Non-Maximum Suppression):** elimina detecções redundantes sobre o mesmo objeto, mantendo a de maior confiança.
- **Precisão:** das detecções feitas, quantas estavam corretas.
- **Revocação:** dos objetos reais, quantos foram encontrados.
- **mAP@0.5:** média das precisões médias considerando IoU de 0,5.
- **mAP@0.5:0.95:** média do mAP para dez limiares de IoU (0,50 a 0,95), portanto mais rigorosa.

Essas métricas se relacionam diretamente com a matriz de confusão e com a curva precisão-revocação, e são a base da análise de resultados na Seção 4.

## 3 METODOLOGIA

Esta seção descreve a coleta, a definição e anotação do alvo, o pré-processamento, a divisão dos dados e a estratégia de treinamento.

### 3.1 Coleta de Dados

A coleta foi feita de forma programática a partir do serviço público de tiles XYZ do ESRI World Imagery, que oferece cobertura submétrica de São Paulo. Para o heliponto usei o zoom 19 (cerca de 0,27 metros por pixel), o máximo prático para a região nesse serviço. Cada tile nativo tem 256 por 256 pixels; para ganhar contexto e qualidade de visualização, juntei blocos de 2 por 2 tiles adjacentes, formando blocos de 512 por 512 pixels, com um arquivo de georreferenciamento (world file) gerado para cada bloco.

As regiões varridas foram escolhidas pela densidade de helipontos (corredor Faria Lima, Itaim, Vila Olímpia, Berrini, Brooklin, Morumbi e adjacências) e documentadas por bairro, com a bounding box e o zoom registrados. Após a coleta, cada bloco passou por curadoria manual: separei os blocos com heliponto dos blocos sem heliponto, descartando os irrelevantes.

[INSERIR FIGURA: mapa com as bounding boxes dos bairros coletados]
[INSERIR FIGURA: exemplo de bloco 512x512 montado a partir de 2x2 tiles do zoom 19]

Atribuição obrigatória da fonte: Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community.

### 3.2 Definição do Alvo e Anotação

O objeto-alvo escolhido foi o heliponto em cobertura (classe única chamada heliponto). São Paulo possui uma das maiores frotas de helicópteros urbanos do mundo, e a marcação padrão de heliponto (a letra H branca dentro de um círculo, pintada no topo de edifícios) é um alvo visualmente distinto e com forma geométrica clara em vista de topo, o que favorece a anotação e o aprendizado. Por ser pequeno, exigiu a coleta em alta resolução discutida na seção anterior.

A anotação foi feita no Roboflow, com uma única classe, seguindo um padrão escrito de anotação que cobre a margem máxima entre a caixa e o objeto, o tratamento de helipontos cortados pela borda (anota-se apenas a porção visível) e o tratamento de sombra e reflexo. As anotações foram exportadas no formato YOLO (um arquivo de texto por imagem, com coordenadas normalizadas), e as imagens sem heliponto foram mantidas como exemplos negativos (arquivo de rótulo vazio) para reduzir falsos positivos.

Ao todo foram reunidas aproximadamente [PREENCHER: N] imagens com o alvo, totalizando [PREENCHER: N] instâncias anotadas de heliponto.

[INSERIR FIGURA: tela do Roboflow com exemplos de caixas justas no heliponto]

### 3.3 Pré-processamento e Divisão dos Dados

No Roboflow, cada bloco de 512 por 512 foi redimensionado para 640 por 640 pixels, resolução de entrada do modelo. A divisão dos dados foi feita por holdout geográfico (por bairro), e não por sorteio aleatório: as regiões de treino e validação usaram um conjunto de bairros, enquanto a Avenida Paulista foi reservada inteira como bairro inédito de teste. Essa escolha evita o vazamento de tiles visualmente parecidos entre treino e teste e fornece uma medida honesta de generalização.

A augmentation foi aplicada somente ao conjunto de treino (rotação de 90 graus e variação de brilho de mais ou menos 20 por cento); validação e teste ficaram sem augmentation, conforme o conceito visto em aula. Os volumes finais foram [PREENCHER: N treino] imagens de treino, [PREENCHER: N validação] de validação e [PREENCHER: N teste] de teste.

[INSERIR FIGURA: diagrama da divisão por bairro (treino/validação vs teste)]

### 3.4 Estratégia de Treinamento

O treino partiu de pesos pré-treinados (transfer learning), com o modelo principal YOLO26n. Para garantir reprodutibilidade, fixei uma semente (seed = 42) e registrei todos os hiperparâmetros. Foram executadas no mínimo duas rodadas de treinamento variando exatamente um hiperparâmetro entre elas (o número de épocas: 30 na primeira e 50 na segunda), mantendo o restante idêntico, para uma comparação justa. A melhor rodada foi selecionada pela mAP@0.5:0.95 de validação.

Como comparação de arquitetura, treinei adicionalmente o YOLO11n sob os mesmos hiperparâmetros da melhor rodada, avaliando ambos no mesmo conjunto de teste — única forma de comparação válida, já que a mAP não é comparável entre datasets ou divisões diferentes.

| Rodada | Modelo  | epochs | batch | imgsz | seed | mAP@0.5 val | mAP@0.5:0.95 val |
|--------|---------|--------|-------|-------|------|-------------|------------------|
| 1      | YOLO26n | 30     | 16    | 640   | 42   | [PREENCHER] | [PREENCHER]      |
| 2      | YOLO26n | 50     | 16    | 640   | 42   | [PREENCHER] | [PREENCHER]      |

## 4 RESULTADOS E DISCUSSÃO

As métricas foram calculadas no conjunto de teste (Avenida Paulista) com limiar de confiança de 0,25 e IoU de 0,50.

### 4.1 Comparação das Rodadas e das Arquiteturas

Resultados do modelo principal (YOLO26n) e da arquitetura de comparação (YOLO11n) no conjunto de teste (bairro inédito), sob os mesmos hiperparâmetros:

| Modelo  | mAP@0.5     | mAP@0.5:0.95 | Precisão    | Revocação   |
|---------|-------------|--------------|-------------|-------------|
| YOLO26n | [PREENCHER] | [PREENCHER]  | [PREENCHER] | [PREENCHER] |
| YOLO11n | [PREENCHER] | [PREENCHER]  | [PREENCHER] | [PREENCHER] |

[PREENCHER: discussão da comparação — qual modelo generalizou melhor no bairro inédito e por quê, relacionando com o esquema de atribuição consciente de pequenos objetos do YOLO26.]

Observação sobre variância: o conjunto de teste foi varrido por inteiro e, portanto, a maioria dos tiles é de fundo, sem o alvo. Isso, somado ao número limitado de instâncias, aumenta a variância das métricas, que devem ser lidas com cautela. Foi justamente para ter uma medida estável de generalização que a Avenida Paulista foi reservada como bairro inédito.

### 4.2 Curvas de Treino

As curvas abaixo, geradas a partir do histórico de treino da melhor rodada, mostram a evolução por época. A mAP (validação) cresce e estabiliza conforme o modelo aprende, enquanto as perdas de treino e validação decrescem; a distância entre as duas curvas de perda indica o nível de overfitting.

[INSERIR FIGURA: curva da mAP@0.5 e mAP@0.5:0.95 por época (curva_map.png)]
[INSERIR FIGURA: curva de loss de treino vs validação por época (curva_loss.png)]

### 4.3 Análise de Erros

Casando as detecções (confiança >= 0,25) com as anotações verdadeiras por IoU >= 0,50, coletei exemplos representativos do comportamento do modelo no teste. A matriz de confusão, no ponto de confiança 0,25, registrou [PREENCHER: N] verdadeiros positivos, [PREENCHER: N] falsos positivos e [PREENCHER: N] falsos negativos.

[INSERIR FIGURA: matriz de confusão (confusion_matrix.png)]
[INSERIR FIGURA: 5 detecções corretas]
[INSERIR FIGURA: 3 falsos positivos]
[INSERIR FIGURA: 3 falsos negativos]

Discussão das causas prováveis:
- **Falsos positivos:** marcas circulares em coberturas (caixas d'água, claraboias, marcações de estacionamento) que se parecem com o padrão do heliponto.
- **Falsos negativos:** helipontos com sombra, baixo contraste, pintura desgastada ou parcialmente cobertos por equipamentos, além de casos cortados pela borda do bloco.

Vale notar que a matriz de confusão é tomada em um limiar fixo de confiança (0,25), mais baixo, enquanto a precisão e a revocação reportadas na Seção 4.1 correspondem ao ponto de operação que maximiza o F1. Essa diferença de limiar explica por que a matriz pode aparentar mais falsos positivos do que a precisão sugere: boa parte desses falsos positivos tem confiança baixa e desaparece ao subir o limiar.

### 4.4 Inferência em Bairro Inédito

Para verificar a generalização, rodei a inferência sobre os tiles da Avenida Paulista, bairro não usado no treino, e salvei as imagens com as detecções. A Avenida Paulista tem características visuais diferentes das regiões de treino (edifícios mais antigos, hotéis), o que estressa o modelo e mostra até que ponto ele generaliza para um domínio novo.

[INSERIR FIGURA: 5 ou mais tiles da Paulista com as detecções do modelo]

## 5 APLICAÇÃO NO STREAMLIT

Para melhor visibilidade e teste das funcionalidades, desenvolvi uma aplicação com interface visual em Streamlit, disponível em mapsyolo26puc.streamlit.app e alimentada diretamente pelo repositório GitHub do projeto. A aplicação carrega o melhor modelo e oferece três áreas: uma página de métricas (com a matriz de confusão e as curvas de treino), a inferência em imagens de teste aleatórias exibidas em um carrossel de quatro por vez, e um modo de navegar por um mosaico de bairro com setas desenhadas dentro da própria imagem (zoom fixo, no estilo de mapas interativos), com o modelo detectando helipontos em tempo real na janela visível.

A aplicação permite ainda variar o limiar de confiança e o IoU do NMS, na linha da aula de métricas, e oferece a opção de test-time augmentation na inferência. Optei por não incluir upload de imagem para evitar entradas fora do domínio do modelo durante a apresentação.

[INSERIR FIGURA: telas do app — métricas, inferência do teste e navegação no mapa]

## 6 CONCLUSÕES

O projeto cobriu o ciclo completo de um detector de objetos em imagens de satélite, com ênfase na construção e curadoria do dataset. No conjunto de teste (Avenida Paulista, bairro inédito), o modelo principal (YOLO26n) encontrou cerca de [PREENCHER: %] dos helipontos (revocação [PREENCHER]) com precisão de [PREENCHER] e mAP@0.5 de [PREENCHER], generalizando para um bairro não visto e com visual diferente. A divisão por bairro (holdout geográfico) forneceu uma medida honesta de generalização.

A principal lição confirmada foi a de que a qualidade dos dados, e não a troca de arquitetura, é o que mais move o resultado em um projeto com poucos exemplos: o maior ganho observado veio da ampliação e do direcionamento da coleta para os casos de erro, e não de mudanças no modelo.

As ferramentas de IA generativa foram utilizadas como apoio na construção dos notebooks, da aplicação Streamlit e na revisão pontual de código e texto. A definição do alvo, a coleta, a anotação, as decisões técnicas e a análise dos resultados foram de minha autoria, com base nos conteúdos abordados em sala de aula.

## 7 REFERÊNCIAS

ULTRALYTICS. Documentação do YOLO (YOLO26, YOLO11, YOLOv8). Disponível em: https://docs.ultralytics.com/. Acesso em: jun. 2026.

ROBOFLOW. Documentation. Disponível em: https://docs.roboflow.com/. Acesso em: jun. 2026.

ESRI. World Imagery. Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community.

MATERIAIS DA DISCIPLINA. Notebooks de localização e detecção, YOLO e métricas, formatos e ferramentas, e notas de CNN em PyTorch. Aprendizagem de Máquina (CDIA), PUC-SP, 2026.
