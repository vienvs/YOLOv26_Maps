# Guia do código — para apresentar com domínio

Este guia explica, em linguagem simples, o que cada parte dos notebooks faz e por quê.
A ideia é que você consiga explicar qualquer célula na banca. São dois notebooks:
o de **coleta** e o de **treino/avaliação**.

---

# Parte A — Notebook de coleta (coleta_helipontos.ipynb)

Objetivo: baixar as imagens de satélite e montar os blocos que depois você anota.

1. **Imports e endpoint do ESRI**
   - Importa `requests` (baixar da internet) e `PIL` (montar/abrir imagens).
   - Define a URL dos tiles do ESRI World Imagery e um cabeçalho de uso educacional.

2. **Converter bairro em tiles (deg2tile)**
   - Você dá um retângulo do bairro (bounding box: lon/lat mínimos e máximos).
   - A função transforma latitude/longitude em índices de tile (x, y) para um zoom.
   - Intuição: o mundo é dividido em uma grade de quadradinhos; essa conta descobre
     quais quadradinhos caem dentro do bairro.

3. **Baixar os tiles**
   - Um laço percorre todos os (x, y) do bairro e baixa cada tile de 256x256.
   - Tem retry (tenta de novo se falhar) e um filtro de tamanho mínimo para descartar
     tile "vazio" (placeholder).

4. **Montar blocos 512x512 (2x2 tiles) + world file**
   - Junta 4 tiles vizinhos em uma imagem maior, para dar mais contexto ao modelo.
   - Gera um `.jgw` (world file): um arquivo de texto que guarda a referência
     geográfica do bloco (útil para saber onde no mapa aquele bloco está).

5. **Curadoria**
   - Você revisa os blocos e separa os que têm heliponto dos que não têm.
   - Frase para a banca: "a coleta é automática, mas a curadoria é manual — e é parte
     do esforço de qualidade dos dados."

Conceito-chave para defender: **zoom 19 (~0,27 m/pixel)** foi escolhido porque
heliponto é pequeno; em zoom menor ele quase some.

---

# Parte B — Notebook de treino/avaliação (treino_avaliacao.ipynb)

Objetivo: treinar o YOLO, avaliar e analisar erros. É o coração do projeto.

## 1. Instalação e imports
- `!pip install ultralytics roboflow` instala a biblioteca do YOLO e o cliente do Roboflow.
- Monta o Google Drive (`drive.mount`) para salvar pesos e resultados sem perder ao
  fechar a sessão do Colab.
- `PASTA_BASE` é a pasta no Drive onde tudo é salvo.

## 2. Baixar o dataset do Roboflow
- A chave de API é pedida com `getpass` (não fica escrita no código — boa prática de
  segurança).
- `projeto.version(N).download('yolo26')` baixa a versão fixa do dataset já no formato
  YOLO. Usar uma versão fixa garante **reprodutibilidade** (todo mundo baixa o mesmo).
- `caminho_data` aponta para o `data.yaml`, o arquivo que diz onde estão treino/val/
  teste e quais são as classes.

## 3. Validar o dataset
- `validar_yaml` confere que há **exatamente 1 classe** (nc = 1, o heliponto).
- `conferir_rotulos` checa se as coordenadas das caixas estão entre 0 e 1 (formato
  YOLO normalizado).
- Para a banca: "essa checagem evita treinar com rótulo quebrado."

## 4. Função de treino (transfer learning)
- `YOLO('yolo26n.pt')` carrega o modelo **já pré-treinado** (transfer learning): ele
  já sabe enxergar formas gerais; a gente só ensina o heliponto.
- `.train(data=..., epochs=..., imgsz=640, batch=16, seed=42, ...)` treina.
- `fixar_seed` + `seed=42`: garante que rodar de novo dá quase o mesmo resultado
  (**reprodutibilidade**).
- Salva tudo em `runs/detect/<nome_da_rodada>` (inclui o `best.pt`, os gráficos e a
  matriz de confusão, porque `plots=True`).

## 5. Duas rodadas (experimento controlado)
- Rodada 1: 30 épocas. Rodada 2: 50 épocas. **Só muda 1 hiperparâmetro** (épocas),
  todo o resto igual.
- Para a banca: "mudar só uma variável por vez é o que torna a comparação justa."

## 6. Escolher a melhor rodada
- Compara as rodadas pela **mAP@0.5:0.95 de validação** e escolhe o melhor `best.pt`.
- Intuição: a validação é o "simulado"; escolho o checkpoint que foi melhor nele.

## 7. Avaliar no teste (bairro inédito)
- `model.val(data=..., split='test', conf=0.25, iou=0.50)` roda no conjunto de teste
  (a Paulista, que o modelo nunca viu).
- Imprime **mAP@0.5, mAP@0.5:0.95, precisão e revocação**.
- Para a banca, saiba dizer cada uma (ver glossário no fim).

## 8. Salvar a matriz de confusão e as métricas
- Copia `confusion_matrix.png` para o Drive e para a pasta do app.
- Salva `metricas.json` (números que o app e o relatório consomem).

## 9. Gerar as curvas (curva_map.png e curva_loss.png)
- Lê o `results.csv` (histórico de cada época) e desenha:
  - mAP@0.5 e mAP@0.5:0.95 subindo por época.
  - loss de treino vs validação descendo por época.
- A média móvel (suavização) só deixa a tendência mais limpa, sem esconder os dados.

## 10. Análise de erros (5 acertos, 3 FP, 3 FN)
- `matriz_iou`: calcula a sobreposição (IoU) entre cada caixa prevista e cada caixa
  verdadeira.
- `contar_tp_fp_fn`: usando IoU >= 0,5, conta acertos (TP), falsos positivos (FP) e
  falsos negativos (FN).
- `carregar_gt`: lê os rótulos verdadeiros (aceita tanto caixa quanto polígono).
- O laço percorre o teste e coleta exemplos de cada tipo para mostrar no relatório.
- Para a banca: "essa parte mostra ONDE o modelo erra, não só quanto ele erra."

## 11. Inferência em bairro inédito
- `model.predict(source='imagens_ineditas/', conf=0.25, save=True)` roda nos tiles da
  Paulista e salva as imagens com as caixas desenhadas.
- É a prova visual de que o modelo generaliza para uma região nova.

## 12. Comparação de arquitetura (YOLO11n)
- Treina o YOLO11n com os **mesmos hiperparâmetros** e avalia no **mesmo teste**.
- Só assim a comparação é válida (mAP não se compara entre datasets diferentes).

---

# Glossário rápido (decore para a banca)

- **Transfer learning:** começar de um modelo já treinado em milhões de imagens e só
  ajustar para a sua tarefa. Ajuda muito quando você tem poucos dados.
- **IoU:** o quanto a caixa prevista se sobrepõe à verdadeira (0 a 1). Critério para
  dizer se a detecção "casou" com o objeto real.
- **NMS:** remove caixas duplicadas sobre o mesmo objeto, mantendo a de maior confiança.
- **Precisão:** das caixas que o modelo disse "heliponto", quantas estavam certas.
- **Revocação:** dos helipontos reais, quantos o modelo achou.
- **mAP@0.5:** qualidade geral considerando IoU de 0,5. Não depende do limiar de confiança.
- **mAP@0.5:0.95:** o mesmo, mas exigindo encaixe cada vez mais perfeito (mais rigoroso).
- **Holdout geográfico:** separar um bairro inteiro para teste, em vez de sortear
  imagens. Evita "cola" (vazamento) entre treino e teste.
- **Augmentation:** criar variações das imagens de treino (rotação, brilho) para o
  modelo generalizar melhor. Só no treino, nunca no teste.
- **Exemplos negativos:** imagens sem heliponto, com rótulo vazio, para o modelo
  aprender a não inventar detecção.
- **Limiar de confiança:** "quão certo" o modelo precisa estar para mostrar a caixa.
  Subir o limiar aumenta precisão e reduz revocação; descer faz o contrário.

# Perguntas que a banca pode fazer (e a resposta curta)

- "Por que YOLO e não uma CNN de classificação?" → Porque preciso localizar (caixa),
  não só dizer se tem ou não; detecção responde "o quê" e "onde".
- "Por que o split é por bairro?" → Para não vazar tiles parecidos entre treino e
  teste; mede generalização de verdade.
- "Por que a validação dá ~0,97 e o teste ~0,75?" → A validação é do mesmo conjunto
  de bairros do treino; o teste é um bairro novo (domínio diferente). O gap é a
  generalização real.
- "A precisão não está baixa?" → É um ponto de operação; subindo o limiar de confiança
  ela sobe. O mAP, que não depende do limiar, mostra que a qualidade geral é estável.
- "Como melhoraria?" → Mais dados anotados (maior alavanca), não trocar de arquitetura.
