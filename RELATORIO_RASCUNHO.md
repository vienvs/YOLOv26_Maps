# Deteccao de Helipontos em Imagens de Satelite com YOLO

Projeto P2 - Aprendizagem de Maquina (CDIA, PUC-SP) - Prof. Dr. Rooney Coelho

Observacao para a montagem final: este e um rascunho de texto. Onde aparecer [INSERIR IMAGEM: ...]
coloque a figura indicada; onde aparecer [VALOR: ...] substitua pelo numero real obtido no notebook.
A atribuicao "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community" deve aparecer em
toda figura que mostre tiles ou mosaicos do ESRI.

## 1. Identificacao do grupo

| # | Nome completo | Matricula |
|---|---------------|-----------|
| 1 |               |           |
| 2 |               |           |
| 3 |               |           |
| 4 |               |           |
| 5 |               |           |

## 2. Introducao

A visao computacional baseada em aprendizado profundo permite localizar e classificar objetos em imagens.
Neste projeto construimos, de ponta a ponta, um detector de um unico objeto em imagens de satelite da
cidade de Sao Paulo: o heliponto em cobertura. O trabalho cobre todo o ciclo de um sistema real, do dado
bruto ao modelo treinado e avaliado, com enfase na construcao do conjunto de dados, que e onde esta a
maior parte do esforco de um projeto de IA.

A mensagem central da disciplina orienta o projeto: cerca de 80 por cento do esforco esta nos dados, e nao
na arquitetura. Como o modelo de deteccao ja vem pre-treinado, o que diferencia um resultado bom de um ruim
e a qualidade da coleta, da curadoria e da anotacao.

## 3. Objetivo

Treinar e avaliar um detector capaz de identificar helipontos em imagens de satelite de Sao Paulo,
medindo a sua capacidade de generalizar para uma regiao nao vista durante o treino. Os objetivos
especificos sao: coletar imagens de forma programatica, anotar com consistencia, treinar com transfer
learning, avaliar com metricas padronizadas e analisar os erros do modelo.

## 4. Definicao do objeto-alvo

O objeto-alvo escolhido foi o heliponto em cobertura (classe unica chamada heliponto). Sao Paulo possui
a maior frota de helicopteros urbanos do mundo, e a marcacao padrao de heliponto (a letra H branca dentro
de um circulo, pintada no topo de edificios) e um alvo visualmente distinto e quase exclusivo da cidade.
O alvo tem forma geometrica clara em vista de topo, o que favorece a anotacao e o aprendizado. Por ser
pequeno (poucos metros), foi necessario coletar em alta resolucao (zoom 19, cerca de 0,27 metros por
pixel).

[INSERIR IMAGEM: exemplo de heliponto em vista de satelite, com a caixa de anotacao]

## 5. Coleta de dados

A coleta foi feita de forma programatica a partir do servico publico de tiles XYZ do ESRI World Imagery,
que oferece cobertura submetrica de Sao Paulo. Para o heliponto usamos o zoom 19 (cerca de 0,27 metros por
pixel), o maximo pratico para a regiao nesse servico. Cada tile nativo tem 256 por 256 pixels; para ganhar
contexto e qualidade de visualizacao, juntamos 2 por 2 tiles adjacentes formando blocos nativos de 512 por
512 pixels, seguindo a mesma ideia do notebook de apoio de alta resolucao.

A coleta foi documentada por bairro, com a bounding box e o zoom registrados. As regioes varridas foram
escolhidas pela densidade de helipontos (corredor Faria Lima, Itaim, Vila Olimpia, Berrini, Brooklin,
Morumbi e adjacencias) e pela proximidade com a regiao da PUC-SP.

[INSERIR IMAGEM: mapa com as bounding boxes dos bairros coletados]
[INSERIR IMAGEM: exemplo de bloco 512x512 montado a partir de 2x2 tiles do zoom 19]

Atribuicao obrigatoria da fonte: Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community.

Apos a coleta, cada bloco passou por curadoria manual: foram separados os blocos com heliponto dos blocos
sem heliponto, descartando os irrelevantes. A documentacao da coleta (bairro, zoom, integrante responsavel,
numero de blocos gerados e aprovados) esta registrada na planilha do grupo.

[INSERIR IMAGEM: trecho da planilha de documentacao da coleta]

## 6. Anotacao

A anotacao foi feita no Roboflow, com uma unica classe (heliponto). Cada integrante anotou uma parcela das
imagens, seguindo um padrao escrito de anotacao definido pelo grupo, que cobre: a margem maxima entre a
caixa e o objeto, o tratamento de helipontos cortados pela borda (anota-se apenas a porcao visivel) e o
tratamento de sombra e reflexo. As anotacoes foram exportadas no formato YOLO (um arquivo de texto por
imagem, com coordenadas normalizadas), e as imagens sem heliponto foram mantidas como exemplos negativos
(arquivo de rotulo vazio) para reduzir falsos positivos.

[INSERIR IMAGEM: tela do Roboflow com exemplos de caixas justas no heliponto]

Ao todo foram reunidas aproximadamente 150 imagens com o alvo, totalizando 201 instancias anotadas de
heliponto. Parte das imagens foi complementada com capturas do Google Earth Web, para aproximar a meta de
instancias e trazer angulos e datas diferentes. O conjunto de teste (Avenida Paulista) contem 33 instancias
de heliponto distribuidas em 195 imagens (a maioria de fundo, sem o alvo).

## 7. Pre-processamento e divisao dos dados

No Roboflow, cada bloco de 512 por 512 foi redimensionado para 640 por 640 pixels, resolucao de entrada do
modelo. A divisao dos dados foi feita por holdout geografico (por bairro), e nao por sorteio aleatorio:
as regioes de treino e validacao usaram um conjunto de bairros, enquanto a Avenida Paulista foi reservada
inteira como bairro inedito de teste. Essa escolha evita o vazamento de tiles visualmente parecidos entre
treino e teste e fornece uma medida honesta de generalizacao.

A augmentation foi aplicada somente ao conjunto de treino (rotacao de 90 graus e variacao de brilho de mais
ou menos 20 por cento); validacao e teste ficaram sem augmentation, conforme o conceito visto em aula. Os
volumes finais foram 419 imagens de treino, 20 de validacao e 195 de teste.

[INSERIR IMAGEM: diagrama da divisao por bairro (treino/validacao vs teste)]

## 8. Metodologia de treinamento

O treino partiu de pesos pre-treinados (transfer learning), com o modelo principal YOLO26n e a opcao de
fallback para YOLOv8n ou YOLO11n. O YOLO26n foi escolhido por trazer melhorias uteis para alvos pequenos,
como o esquema de atribuicao de rotulos consciente de pequenos objetos, alem de inferencia sem supressao
nao maxima explicita e cabeca de deteccao mais leve. A variante nano cabe com folga na GPU T4 do Google
Colab e treina rapido para um dataset pequeno.

Para garantir reprodutibilidade, fixamos uma seed e registramos todos os hiperparametros. Foram executadas
no minimo duas rodadas de treinamento variando exatamente um hiperparametro entre elas (o numero de epocas:
30 na primeira e 50 na segunda), mantendo o restante identico, para uma comparacao justa.

| Rodada | epochs | batch | imgsz | seed | mAP@0.5 val | mAP@0.5:0.95 val |
|--------|--------|-------|-------|------|-------------|------------------|
| 1      | 30     | 16    | 640   | 42   | [PREENCHER: valor impresso pela celula de comparacao] | [PREENCHER] |
| 2      | 50     | 16    | 640   | 42   | 0,987       | 0,786            |

Na validacao, o YOLO26n atingiu mAP@0.5 de 0,987 e mAP@0.5:0.95 de 0,786 (precisao 1,000 e revocacao
0,936 sobre 17 instancias). Como a validacao tem apenas 20 imagens, esses valores sao otimistas e devem ser
lidos junto com os resultados no teste (bairro inedito), que medem a generalizacao real.

[INSERIR IMAGEM: curvas de perda e metricas ao longo das epocas (results.png)]

## 9. Avaliacao

As metricas foram calculadas no conjunto de teste (Avenida Paulista) com limiar de confianca de 0,25 e IoU
de 0,50. Relembrando as definicoes vistas em aula: a precisao mede, das deteccoes feitas, quantas estavam
certas; a revocacao mede, dos alvos reais, quantos foram encontrados; o mAP@0.5 e a media das precisoes
medias com IoU de 0,5; e o mAP@0.5:0.95 e a media para varios limiares de IoU, sendo mais rigoroso.

Resultados do modelo principal (YOLO26n) no conjunto de teste (Avenida Paulista):

| Metrica | Valor (teste) |
|---------|---------------|
| mAP@0.5 | 0,723 |
| mAP@0.5:0.95 | 0,512 |
| Precisao | 0,812 |
| Revocacao | 0,786 |

As metricas acima sao as reportadas pelo model.val no conjunto de teste. A matriz de confusao, no ponto de
confianca 0,25, registrou 27 verdadeiros positivos, 10 falsos positivos e 6 falsos negativos.

[INSERIR IMAGEM: matriz de confusao (confusion_matrix.png)]

Comparacao de arquitetura no teste (alem das duas rodadas obrigatorias), sob os mesmos hiperparametros:

| Modelo  | mAP@0.5 | mAP@0.5:0.95 | Precisao | Revocacao |
|---------|---------|--------------|----------|-----------|
| YOLO26n | 0,723 | 0,512 | 0,812 | 0,786 |
| YOLO11n | 0,635   | 0,441        | 0,76     | 0,576     |

O YOLO26n obteve revocacao bem maior no bairro inedito (0,82 contra 0,576 do YOLO11n), ou seja, encontrou
muito mais helipontos numa regiao nao vista. Isso e coerente com o esquema de atribuicao consciente de
pequenos objetos do YOLO26, util para um alvo pequeno como o heliponto. A precisao ficou proxima entre os
dois modelos.

[INSERIR IMAGEM: matriz de confusao (confusion_matrix.png)]

Observacao sobre variancia: a validacao tem apenas 20 imagens e o teste foi varrido por inteiro (a maioria
dos tiles e fundo, sem heliponto). Ambos aumentam a variancia das metricas e devem ser lidos com cautela.

## 10. Analise de erros

Apresentamos exemplos representativos do comportamento do modelo no teste.

[INSERIR IMAGEM: 5 deteccoes corretas]
[INSERIR IMAGEM: 3 falsos positivos]
[INSERIR IMAGEM: 3 falsos negativos]

Discussao das causas provaveis:
- Falsos positivos: marcas circulares em coberturas (caixas d agua, claraboias, marcacoes de estacionamento)
  que se parecem com o padrao do heliponto.
- Falsos negativos: helipontos com sombra, baixo contraste, pintura desgastada ou parcialmente cobertos por
  equipamentos, alem de casos cortados pela borda do bloco.

## 11. Inferencia em bairro inedito

Para verificar a generalizacao, rodamos a inferencia sobre os tiles da Avenida Paulista, bairro nao usado no
treino, e salvamos as imagens com as deteccoes. Discutimos a seguir pelo menos cinco tiles.

[INSERIR IMAGEM: 5 ou mais tiles da Paulista com as deteccoes do modelo]

Comentario: a Avenida Paulista tem caracteristicas visuais diferentes das regioes de treino (edificios mais
antigos, hoteis), o que estressa o modelo e mostra ate que ponto ele generaliza para um dominio novo.

## 12. Aplicacao web (opcional)

Como diferencial, desenvolvemos uma aplicacao em Streamlit que carrega o melhor modelo e oferece: uma pagina
de metricas, inferencia por upload de imagem, inferencia nas imagens do teste e um modo de navegar por um
mosaico de bairro com as setas (zoom fixo, no estilo de mapas interativos), com o modelo detectando
helipontos em tempo real na janela visivel.

[INSERIR IMAGEM: telas do app (metricas, inferencia e modo de navegacao no mapa)]

## 13. Melhorias e trabalhos futuros (com base nos conteudos de aula)

As melhorias abaixo se apoiam diretamente em topicos vistos na disciplina:
- Ajuste do limiar de confianca e do IoU do NMS para equilibrar precisao e revocacao, conforme a aula de
  metricas (IoU, NMS, curva precisao-revocacao). A aplicacao ja permite variar esses limiares.
- Test-time augmentation na inferencia, na linha do uso de augmentation visto em aula, para tentar recuperar
  helipontos dificeis (mais lento, mas pode aumentar a revocacao).
- Treino com precisao mista (AMP) para acelerar e economizar memoria na T4, conforme a aula de treinamento.
- Early stopping por paciencia para evitar overfitting quando a perda de validacao para de melhorar.
- Visualizacao de mapas de ativacao e de filtros (e a ideia do Grad-CAM, vista nas notas de CNN) para
  entender onde a rede esta olhando e diagnosticar falsos positivos (shortcut features).
- Inferencia por janela deslizante sobre um mosaico grande, reaproveitada no modo de navegacao do app, para
  cobrir um bairro inteiro mantendo o tamanho de entrada do modelo.
- Mais dados: como heliponto e um alvo raro, ampliar a coleta para mais bairros e datas tende a trazer mais
  ganho do que trocar a arquitetura, reforcando a licao de que o dado importa mais.

## 14. Limitacoes

- Volume de positivos limitado pela raridade do heliponto na resolucao de 0,27 metros por pixel.
- Validacao pequena (20 imagens) e teste majoritariamente de fundo, aumentando a variancia das metricas.
- A imagem do ESRI tem alguns anos; prediais muito recentes podem nao aparecer.
- O modo de navegacao do app usa botoes de seta (a captura continua de teclado e uma melhoria futura).

## 15. Uso de IA generativa

Declarar aqui qual ferramenta de IA foi usada, em que etapas (por exemplo, apoio na estruturacao do codigo
dos notebooks, no app e na redacao deste relatorio) e o que foi efetivamente aproveitado e revisado pelo
grupo.

## 16. Conclusao

O projeto cobriu o ciclo completo de um detector de objetos em imagens de satelite, com enfase na construcao
e curadoria do dataset. No conjunto de teste (Avenida Paulista, bairro inedito) o modelo principal (YOLO26n)
encontrou cerca de 79 por cento dos helipontos (revocacao 0,786) com precisao de 0,812 e mAP@0.5 de 0,723,
generalizando bem para um bairro nao visto e com visual diferente. A comparacao com o YOLO11n reforcou a
escolha do YOLO26n, que teve revocacao bem maior no bairro inedito (0,786 contra 0,576). A divisao por bairro
(holdout geografico) forneceu uma medida honesta de generalizacao. A principal licao confirmada foi a de que
a qualidade dos dados, e nao a troca de arquitetura, e o que mais move o resultado em um projeto com poucos
exemplos.

## 17. Referencias

- Documentacao da Ultralytics (YOLO26, YOLOv8, YOLO11).
- Documentacao do Roboflow.
- Materiais da disciplina (notebooks de localizacao/deteccao, YOLO e metricas, formatos e ferramentas, e
  notas de CNN em PyTorch).
- Fonte das imagens: Esri, Maxar, Earthstar Geographics, and the GIS User Community.
