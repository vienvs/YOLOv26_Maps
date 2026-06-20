import os
import json

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# aplicacao de demonstracao do detector de helipontos
# mostra metricas, inferencia em imagens do teste, inferencia por upload
# e um modo de navegar por um mosaico de bairro com inferencia em tempo real
# atribuicao obrigatoria da fonte das imagens:
# Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community

ATRIBUICAO = "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"

# resolve os caminhos a partir da pasta do proprio script
# assim funciona tanto local quanto no streamlit cloud (onde o diretorio de trabalho e a raiz do repo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_BEST = os.path.join(BASE_DIR, "best.pt")
CAMINHO_METRICAS = os.path.join(BASE_DIR, "metricas.json")
CAMINHO_MATRIZ = os.path.join(BASE_DIR, "confusion_matrix.png")
PASTA_TESTE = os.path.join(BASE_DIR, "test", "images")
CAMINHO_MOSAICO = os.path.join(BASE_DIR, "mosaico_bairro.png")

# tamanho da janela de visualizacao do mapa (sem alterar o zoom)
TAMANHO_JANELA = 640
PASSO = 320


@st.cache_resource
def carregar_modelo(caminho):
    # carrega os pesos do modelo uma unica vez e mantem em cache
    return YOLO(caminho)


@st.cache_data
def carregar_mosaico(caminho):
    # carrega o mosaico do bairro uma unica vez
    return Image.open(caminho).convert("RGB")


def desenhar(modelo, imagem, conf, iou, usar_tta):
    # roda a inferencia e devolve a imagem anotada e o numero de deteccoes
    resultado = modelo.predict(source=imagem, conf=conf, iou=iou, augment=usar_tta, verbose=False)
    anotada = resultado[0].plot()
    # o plot do ultralytics vem em bgr; converte para rgb
    anotada_rgb = anotada[:, :, ::-1]
    n_deteccoes = len(resultado[0].boxes)
    return Image.fromarray(anotada_rgb), n_deteccoes


def listar_imagens(pasta):
    # devolve os caminhos de imagem de uma pasta, em ordem estavel
    arquivos = []
    if not os.path.isdir(pasta):
        return arquivos
    for nome in os.listdir(pasta):
        if nome.endswith(".jpg") or nome.endswith(".png") or nome.endswith(".jpeg"):
            arquivos.append(os.path.join(pasta, nome))
    arquivos.sort()
    return arquivos


st.set_page_config(page_title="Detector de helipontos", layout="wide")
st.title("Detector de helipontos em imagens de satelite")
st.caption(ATRIBUICAO)

st.sidebar.header("parametros de inferencia")
conf = st.sidebar.slider("confianca minima", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("iou do nms", 0.1, 0.9, 0.5, 0.05)
usar_tta = st.sidebar.checkbox("test-time augmentation (mais lento)", value=False)

modelo = None
if os.path.exists(CAMINHO_BEST):
    modelo = carregar_modelo(CAMINHO_BEST)
else:
    st.warning("coloque o arquivo best.pt na mesma pasta do app para habilitar a inferencia")

aba_metricas, aba_upload, aba_teste, aba_mapa = st.tabs(
    ["metricas", "inferencia por upload", "inferencia do teste", "andar pelo mapa"]
)

with aba_metricas:
    st.subheader("metricas de avaliacao no conjunto de teste")
    if os.path.exists(CAMINHO_METRICAS):
        with open(CAMINHO_METRICAS, "r") as arquivo:
            metricas = json.load(arquivo)
        coluna1, coluna2, coluna3, coluna4 = st.columns(4)
        coluna1.metric("mAP@0.5", round(metricas.get("map50", 0.0), 4))
        coluna2.metric("mAP@0.5:0.95", round(metricas.get("map", 0.0), 4))
        coluna3.metric("precisao", round(metricas.get("precisao", 0.0), 4))
        coluna4.metric("revocacao", round(metricas.get("revocacao", 0.0), 4))
    else:
        st.info("gere o arquivo metricas.json no notebook de treino para ver os valores aqui")
    if os.path.exists(CAMINHO_MATRIZ):
        st.image(CAMINHO_MATRIZ, caption="matriz de confusao", width=500)

with aba_upload:
    st.subheader("envie uma imagem de satelite")
    arquivo = st.file_uploader("imagem", type=["png", "jpg", "jpeg"])
    if arquivo is not None and modelo is not None:
        imagem = Image.open(arquivo).convert("RGB")
        anotada, n = desenhar(modelo, imagem, conf, iou, usar_tta)
        st.image(anotada, caption="deteccoes: " + str(n))
        st.caption(ATRIBUICAO)

with aba_teste:
    st.subheader("inferencia nas imagens do conjunto de teste")
    imagens = listar_imagens(PASTA_TESTE)
    if len(imagens) == 0:
        st.info("aponte PASTA_TESTE para a pasta test/images do dataset")
    elif modelo is not None:
        indice = st.slider("imagem", 0, len(imagens) - 1, 0)
        caminho = imagens[indice]
        imagem = Image.open(caminho).convert("RGB")
        anotada, n = desenhar(modelo, imagem, conf, iou, usar_tta)
        st.image(anotada, caption=os.path.basename(caminho) + " - deteccoes: " + str(n))

with aba_mapa:
    st.subheader("andar pelo mapa do bairro (zoom fixo)")
    st.write("use as setas para mover a janela de visualizacao; o modelo detecta helipontos em tempo real")
    if not os.path.exists(CAMINHO_MOSAICO):
        st.info("salve um mosaico de um bairro como mosaico_bairro.png para usar este modo")
    elif modelo is not None:
        mosaico = carregar_mosaico(CAMINHO_MOSAICO)
        largura, altura = mosaico.size
        max_x = max(0, largura - TAMANHO_JANELA)
        max_y = max(0, altura - TAMANHO_JANELA)

        if "pos_x" not in st.session_state:
            st.session_state.pos_x = 0
        if "pos_y" not in st.session_state:
            st.session_state.pos_y = 0

        coluna_esq, coluna_centro, coluna_dir = st.columns(3)
        if coluna_centro.button("cima"):
            st.session_state.pos_y = max(0, st.session_state.pos_y - PASSO)
        if coluna_esq.button("esquerda"):
            st.session_state.pos_x = max(0, st.session_state.pos_x - PASSO)
        if coluna_dir.button("direita"):
            st.session_state.pos_x = min(max_x, st.session_state.pos_x + PASSO)
        if coluna_centro.button("baixo"):
            st.session_state.pos_y = min(max_y, st.session_state.pos_y + PASSO)

        esquerda = st.session_state.pos_x
        topo = st.session_state.pos_y
        janela = mosaico.crop((esquerda, topo, esquerda + TAMANHO_JANELA, topo + TAMANHO_JANELA))
        anotada, n = desenhar(modelo, janela, conf, iou, usar_tta)
        st.image(anotada, caption="posicao x=" + str(esquerda) + " y=" + str(topo) + " - deteccoes: " + str(n))
        st.caption(ATRIBUICAO)
