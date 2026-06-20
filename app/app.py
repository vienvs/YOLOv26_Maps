import os
import json

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# aplicacao de demonstracao do detector de helipontos
# mostra metricas de avaliacao, inferencia nas imagens de teste em tempo real
# e um modo de andar pelo mapa de um bairro com deteccao ao vivo
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

# tamanho de cada bloco (imagem do dataset) e da vista do mapa
TAMANHO_BLOCO = 640


def css_tema(tema):
    # devolve o css da pagina com a fonte inter e as cores do tema claro ou escuro
    if tema == "escuro":
        fundo = "#0e1117"
        fundo_painel = "#161b22"
        texto = "#e6edf3"
        borda = "#30363d"
    else:
        fundo = "#ffffff"
        fundo_painel = "#f5f7fa"
        texto = "#1a1a1a"
        borda = "#e2e6ea"
    estilo = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp, button, input, textarea, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stApp { background-color: FUNDO; color: TEXTO; }
    section[data-testid="stSidebar"] { background-color: FUNDO_PAINEL; }
    section[data-testid="stSidebar"] * { color: TEXTO; }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: TEXTO; }
    [data-testid="stMetric"] {
        background-color: FUNDO_PAINEL;
        border: 1px solid BORDA;
        border-radius: 12px;
        padding: 16px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 8px 16px;
    }
    .stButton button {
        border-radius: 10px;
        border: 1px solid BORDA;
        font-weight: 600;
    }
    .rodape { color: TEXTO; opacity: 0.6; font-size: 0.8rem; margin-top: 24px; }
    </style>
    """
    estilo = estilo.replace("FUNDO_PAINEL", fundo_painel)
    estilo = estilo.replace("FUNDO", fundo)
    estilo = estilo.replace("TEXTO", texto)
    estilo = estilo.replace("BORDA", borda)
    return estilo


@st.cache_resource
def carregar_modelo(caminho):
    # carrega os pesos do modelo uma unica vez e mantem em cache
    return YOLO(caminho)


def listar_imagens(pasta):
    # devolve os caminhos de imagem de uma pasta em ordem estavel
    arquivos = []
    if not os.path.isdir(pasta):
        return arquivos
    for nome in os.listdir(pasta):
        if nome.endswith(".jpg") or nome.endswith(".png") or nome.endswith(".jpeg"):
            arquivos.append(os.path.join(pasta, nome))
    arquivos.sort()
    return arquivos


def parse_xy(nome):
    # extrai os indices x e y do nome bloco_z19_x194190_y297462_...jpg
    base = os.path.basename(nome)
    partes = base.split("_")
    x = None
    y = None
    for parte in partes:
        if parte.startswith("x") and parte[1:].isdigit():
            x = int(parte[1:])
        if parte.startswith("y") and parte[1:].isdigit():
            y = int(parte[1:])
    return x, y


@st.cache_data
def construir_grade(pasta):
    # monta uma grade (coluna, linha) -> caminho a partir dos indices de tile no nome dos blocos
    imagens = listar_imagens(pasta)
    posicoes = {}
    xs = []
    ys = []
    for caminho in imagens:
        x, y = parse_xy(caminho)
        if x is None or y is None:
            continue
        posicoes[(x, y)] = caminho
        xs.append(x)
        ys.append(y)
    if len(posicoes) == 0:
        return {}, 0, 0
    x_min = min(xs)
    y_min = min(ys)
    grade = {}
    colunas = 0
    linhas = 0
    for chave in posicoes:
        x, y = chave
        coluna = (x - x_min) // 2
        linha = (y - y_min) // 2
        grade[(coluna, linha)] = posicoes[chave]
        if coluna > colunas:
            colunas = coluna
        if linha > linhas:
            linhas = linha
    return grade, colunas + 1, linhas + 1


def desenhar(modelo, imagem, conf, iou, usar_tta):
    # roda a inferencia e devolve a imagem anotada e o numero de deteccoes
    resultado = modelo.predict(source=imagem, conf=conf, iou=iou, augment=usar_tta, verbose=False)
    anotada = resultado[0].plot()
    # o plot do ultralytics vem em bgr; converte para rgb
    anotada_rgb = anotada[:, :, ::-1]
    return Image.fromarray(anotada_rgb), len(resultado[0].boxes)


def montar_vista(grade, col, row):
    # junta um bloco 2x2 a partir da posicao atual para dar sensacao de mapa continuo
    vista = Image.new("RGB", (2 * TAMANHO_BLOCO, 2 * TAMANHO_BLOCO), (20, 20, 20))
    for dc in range(2):
        for dr in range(2):
            chave = (col + dc, row + dr)
            if chave in grade:
                bloco = Image.open(grade[chave]).convert("RGB").resize((TAMANHO_BLOCO, TAMANHO_BLOCO))
                vista.paste(bloco, (dc * TAMANHO_BLOCO, dr * TAMANHO_BLOCO))
    return vista


st.set_page_config(page_title="Detector de helipontos", layout="wide")

if "tema" not in st.session_state:
    st.session_state.tema = "escuro"

with st.sidebar:
    st.markdown("### aparencia")
    modo_escuro = st.toggle("modo escuro", value=(st.session_state.tema == "escuro"))
    st.session_state.tema = "escuro" if modo_escuro else "claro"
    st.markdown("### parametros de inferencia")
    conf = st.slider("confianca minima", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("iou do nms", 0.1, 0.9, 0.5, 0.05)
    usar_tta = st.checkbox("test-time augmentation (mais lento)", value=False)

st.markdown(css_tema(st.session_state.tema), unsafe_allow_html=True)

st.title("Detector de helipontos em imagens de satelite")
st.write("Projeto P2 - Aprendizagem de Maquina (CDIA, PUC-SP). Modelo YOLO26n treinado em imagens do ESRI World Imagery.")

modelo = None
if os.path.exists(CAMINHO_BEST):
    modelo = carregar_modelo(CAMINHO_BEST)
else:
    st.warning("coloque o arquivo best.pt na pasta do app para habilitar a inferencia")

aba_metricas, aba_teste, aba_mapa = st.tabs(["metricas", "inferencia do teste", "andar pelo mapa"])

with aba_metricas:
    st.subheader("desempenho no conjunto de teste (Avenida Paulista)")
    if os.path.exists(CAMINHO_METRICAS):
        with open(CAMINHO_METRICAS, "r") as arquivo:
            metricas = json.load(arquivo)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP@0.5", round(metricas.get("map50", 0.0), 4))
        c2.metric("mAP@0.5:0.95", round(metricas.get("map", 0.0), 4))
        c3.metric("precisao", round(metricas.get("precisao", 0.0), 4))
        c4.metric("revocacao", round(metricas.get("revocacao", 0.0), 4))
    else:
        st.info("gere o arquivo metricas.json no notebook de treino para ver os valores aqui")
    if os.path.exists(CAMINHO_MATRIZ):
        st.markdown("#### matriz de confusao")
        st.image(CAMINHO_MATRIZ, width=520)
    st.caption("a Avenida Paulista foi reservada como bairro inedito (holdout geografico) para medir a generalizacao")

with aba_teste:
    st.subheader("inferencia nas imagens de teste em tempo real")
    imagens = listar_imagens(PASTA_TESTE)
    if len(imagens) == 0:
        st.info("nao ha imagens em test/images")
    elif modelo is not None:
        if "indice_teste" not in st.session_state:
            st.session_state.indice_teste = 0
        coluna_ant, coluna_prox, coluna_vazio = st.columns([1, 1, 6])
        if coluna_ant.button("anterior"):
            st.session_state.indice_teste = (st.session_state.indice_teste - 1) % len(imagens)
        if coluna_prox.button("proxima"):
            st.session_state.indice_teste = (st.session_state.indice_teste + 1) % len(imagens)
        indice = st.slider("imagem", 0, len(imagens) - 1, st.session_state.indice_teste)
        st.session_state.indice_teste = indice
        caminho = imagens[indice]
        imagem = Image.open(caminho).convert("RGB")
        anotada, n = desenhar(modelo, imagem, conf, iou, usar_tta)
        st.image(anotada, caption=os.path.basename(caminho))
        st.metric("helipontos detectados", n)

with aba_mapa:
    st.subheader("andar pelo mapa do bairro (zoom fixo)")
    st.write("use as setas para percorrer o bairro; o modelo detecta helipontos ao vivo na area visivel")
    grade, colunas, linhas = construir_grade(PASTA_TESTE)
    if len(grade) == 0:
        st.info("nao consegui montar o mapa a partir dos nomes dos blocos")
    elif modelo is not None:
        if "mapa_col" not in st.session_state:
            st.session_state.mapa_col = 0
        if "mapa_row" not in st.session_state:
            st.session_state.mapa_row = 0
        ce, cc, cd = st.columns(3)
        if cc.button("cima"):
            st.session_state.mapa_row = max(0, st.session_state.mapa_row - 1)
        if ce.button("esquerda"):
            st.session_state.mapa_col = max(0, st.session_state.mapa_col - 1)
        if cd.button("direita"):
            st.session_state.mapa_col = min(colunas - 1, st.session_state.mapa_col + 1)
        if cc.button("baixo"):
            st.session_state.mapa_row = min(linhas - 1, st.session_state.mapa_row + 1)
        col = st.session_state.mapa_col
        row = st.session_state.mapa_row
        vista = montar_vista(grade, col, row)
        anotada, n = desenhar(modelo, vista, conf, iou, usar_tta)
        st.image(anotada, caption="posicao coluna " + str(col) + " linha " + str(row) + " de " + str(colunas) + "x" + str(linhas))
        st.metric("helipontos detectados na area", n)

st.markdown("<div class='rodape'>" + ATRIBUICAO + "</div>", unsafe_allow_html=True)
