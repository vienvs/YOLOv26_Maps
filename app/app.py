import os
import json

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# aplicacao de demonstracao do detector de helipontos
# mostra metricas de avaliacao, inferencia nas imagens de teste em carrossel
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

# tamanho de cada bloco (imagem do dataset) e base da vista do mapa
TAMANHO_BLOCO = 640
IMAGENS_POR_PAGINA = 4


def css_tema(tema):
    # devolve o css da pagina com a fonte inter e as cores do tema claro ou escuro
    if tema == "escuro":
        fundo = "#0e1117"
        painel = "#161b22"
        texto = "#e6edf3"
        borda = "#30363d"
    else:
        fundo = "#ffffff"
        painel = "#f3f5f8"
        texto = "#1a1a1a"
        borda = "#e2e6ea"
    estilo = """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp, .stMarkdown, p, label, h1, h2, h3, h4, h5, h6,
    button, input, textarea, select,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    .stApp { background-color: FUNDO; color: TEXTO; }
    header[data-testid="stHeader"] { background-color: FUNDO; }
    [data-testid="stToolbar"] { background-color: FUNDO; }
    section[data-testid="stSidebar"] { background-color: PAINEL; }
    section[data-testid="stSidebar"] * { color: TEXTO; }
    h1, h2, h3, h4, h5, h6, p, label { color: TEXTO; }
    [data-testid="stMetric"] {
        background-color: PAINEL;
        border: 1px solid BORDA;
        border-radius: 12px;
        padding: 16px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 16px; }
    .stButton button {
        border-radius: 999px;
        border: 1px solid BORDA;
        font-weight: 700;
        font-size: 1.1rem;
        background-color: PAINEL;
        color: TEXTO;
    }
    .rodape { color: TEXTO; opacity: 0.6; font-size: 0.8rem; margin-top: 24px; }
    </style>
    """
    estilo = estilo.replace("PAINEL", painel)
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


def anotar_imagem(modelo, imagem, conf, iou, usar_tta):
    # roda a inferencia e devolve a imagem anotada (rgb) e o numero de deteccoes
    resultado = modelo.predict(source=imagem, conf=conf, iou=iou, augment=usar_tta, verbose=False)
    anotada = resultado[0].plot()
    anotada_rgb = anotada[:, :, ::-1]
    return Image.fromarray(anotada_rgb), len(resultado[0].boxes)


@st.cache_data(show_spinner=False)
def anotar_arquivo(caminho, conf, iou, usar_tta, _modelo):
    # versao em cache para as imagens de teste (a chave ignora o modelo por comecar com _)
    imagem = Image.open(caminho).convert("RGB")
    return anotar_imagem(_modelo, imagem, conf, iou, usar_tta)


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

with st.sidebar:
    st.markdown("### Aparência")
    modo_escuro = st.toggle("Modo escuro", value=True, key="modo_escuro")
    st.markdown("### Parâmetros de inferência")
    conf = st.slider("Confiança mínima", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU do NMS", 0.1, 0.9, 0.5, 0.05)
    usar_tta = st.checkbox("Test-time augmentation (mais lento)", value=False)

tema = "escuro" if modo_escuro else "claro"
st.markdown(css_tema(tema), unsafe_allow_html=True)

st.title("Detector de helipontos em imagens de satélite")
st.write("Projeto P2 - Aprendizagem de Máquina (CDIA, PUC-SP). Modelo YOLO26n treinado em imagens do ESRI World Imagery.")

modelo = None
if os.path.exists(CAMINHO_BEST):
    modelo = carregar_modelo(CAMINHO_BEST)
else:
    st.warning("Coloque o arquivo best.pt na pasta do app para habilitar a inferência.")

aba_metricas, aba_teste, aba_mapa = st.tabs(["Métricas", "Inferência do teste", "Andar pelo mapa"])

with aba_metricas:
    st.subheader("Desempenho no conjunto de teste (Avenida Paulista)")
    if os.path.exists(CAMINHO_METRICAS):
        with open(CAMINHO_METRICAS, "r") as arquivo:
            metricas = json.load(arquivo)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP@0.5", round(metricas.get("map50", 0.0), 4))
        c2.metric("mAP@0.5:0.95", round(metricas.get("map", 0.0), 4))
        c3.metric("Precisão", round(metricas.get("precisao", 0.0), 4))
        c4.metric("Revocação", round(metricas.get("revocacao", 0.0), 4))
    else:
        st.info("Gere o arquivo metricas.json no notebook de treino para ver os valores aqui.")
    if os.path.exists(CAMINHO_MATRIZ):
        st.markdown("#### Matriz de confusão")
        st.image(CAMINHO_MATRIZ, width=520)
    st.caption("A Avenida Paulista foi reservada como bairro inédito (holdout geográfico) para medir a generalização.")

with aba_teste:
    st.subheader("Inferência nas imagens de teste")
    imagens = listar_imagens(PASTA_TESTE)
    if len(imagens) == 0:
        st.info("Não há imagens em test/images.")
    elif modelo is not None:
        total_paginas = (len(imagens) + IMAGENS_POR_PAGINA - 1) // IMAGENS_POR_PAGINA
        if "pagina_teste" not in st.session_state:
            st.session_state.pagina_teste = 0
        c_ant, c_meio, c_prox = st.columns([1, 6, 1])
        if c_ant.button("◀", key="teste_ant"):
            st.session_state.pagina_teste = (st.session_state.pagina_teste - 1) % total_paginas
        if c_prox.button("▶", key="teste_prox"):
            st.session_state.pagina_teste = (st.session_state.pagina_teste + 1) % total_paginas
        c_meio.markdown(
            "<p style='text-align:center;margin-top:8px;'>Página "
            + str(st.session_state.pagina_teste + 1) + " de " + str(total_paginas) + "</p>",
            unsafe_allow_html=True,
        )
        inicio = st.session_state.pagina_teste * IMAGENS_POR_PAGINA
        selecao = imagens[inicio:inicio + IMAGENS_POR_PAGINA]
        grade_cols = st.columns(2)
        posicao = 0
        for caminho in selecao:
            anotada, n = anotar_arquivo(caminho, conf, iou, usar_tta, modelo)
            with grade_cols[posicao % 2]:
                st.image(anotada, use_container_width=True)
                st.caption("Helipontos detectados: " + str(n))
            posicao = posicao + 1

with aba_mapa:
    st.subheader("Andar pelo mapa do bairro (zoom fixo)")
    st.write("Use as setas ao redor da imagem para percorrer o bairro; o modelo detecta helipontos ao vivo.")
    grade, colunas, linhas = construir_grade(PASTA_TESTE)
    if len(grade) == 0:
        st.info("Não consegui montar o mapa a partir dos nomes dos blocos.")
    elif modelo is not None:
        if "mapa_col" not in st.session_state:
            st.session_state.mapa_col = 0
        if "mapa_row" not in st.session_state:
            st.session_state.mapa_row = 0
        col_esq, col_centro, col_dir = st.columns([1, 8, 1])
        with col_centro:
            topo = st.columns([5, 2, 5])
            clicou_cima = topo[1].button("▲", key="mapa_cima")
            painel_imagem = st.empty()
            base = st.columns([5, 2, 5])
            clicou_baixo = base[1].button("▼", key="mapa_baixo")
        with col_esq:
            st.write("")
            st.write("")
            clicou_esq = st.button("◀", key="mapa_esq")
        with col_dir:
            st.write("")
            st.write("")
            clicou_dir = st.button("▶", key="mapa_dir")
        # processa os cliques antes de desenhar para nao haver atraso de um rerun
        if clicou_cima:
            st.session_state.mapa_row = max(0, st.session_state.mapa_row - 1)
        if clicou_baixo:
            st.session_state.mapa_row = min(linhas - 1, st.session_state.mapa_row + 1)
        if clicou_esq:
            st.session_state.mapa_col = max(0, st.session_state.mapa_col - 1)
        if clicou_dir:
            st.session_state.mapa_col = min(colunas - 1, st.session_state.mapa_col + 1)
        col = st.session_state.mapa_col
        row = st.session_state.mapa_row
        vista = montar_vista(grade, col, row)
        anotada, n = anotar_imagem(modelo, vista, conf, iou, usar_tta)
        painel_imagem.image(anotada, use_container_width=True,
                            caption="Posição: coluna " + str(col) + ", linha " + str(row) + " (de " + str(colunas) + "x" + str(linhas) + ")")
        st.metric("Helipontos detectados na área", n)

st.markdown("<div class='rodape'>" + ATRIBUICAO + "</div>", unsafe_allow_html=True)
