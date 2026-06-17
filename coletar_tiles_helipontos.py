import math
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor

# coleta de tiles xyz do esri world imagery para o projeto de helipontos
# alvo: heliponto em cobertura, zoom 19 (cerca de 0,27 metros por pixel)
# atribuicao obrigatoria ao usar os tiles em figuras, slides ou relatorio:
# Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community

URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
       "World_Imagery/MapServer/tile/{z}/{y}/{x}")
HEADERS = {"User-Agent": "PUC-SP-AM-Aula-Educacional/1.0"}

ZOOM = 19
TAMANHO_MINIMO_VALIDO = 2521  # bytes; abaixo disso o tile e tratado como placeholder
PASTA_BASE = "coleta_helipontos"

# bbox no formato (lon_min, lat_min, lon_max, lat_max)
# papel indica o destino no split por bairro: treino_val ou teste
REGIOES = {
    "itaim_faria_lima": {
        "bbox": (-46.694, -23.592, -46.672, -23.572),
        "papel": "treino_val",
    },
    "vila_olimpia": {
        "bbox": (-46.695, -23.604, -46.676, -23.590),
        "papel": "treino_val",
    },
    "brooklin_berrini": {
        "bbox": (-46.703, -23.624, -46.683, -23.606),
        "papel": "treino_val",
    },
    "avenida_paulista": {
        "bbox": (-46.660, -23.572, -46.640, -23.555),
        "papel": "teste",
    },
}


def deg2tile(lat, lon, z):
    # converte latitude e longitude em graus para indice de tile x, y no zoom z
    # segue a projecao web mercator usada pelos tiles xyz
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def baixar_tile(z, x, y, pasta_saida):
    # baixa um unico tile com ate 3 tentativas e filtra placeholders pequenos
    destino = os.path.join(pasta_saida, "tile_z" + str(z) + "_x" + str(x) + "_y" + str(y) + ".jpg")
    if os.path.exists(destino):
        return True
    tentativa = 0
    while tentativa < 3:
        try:
            resposta = requests.get(URL.format(z=z, x=x, y=y), headers=HEADERS, timeout=20)
            if resposta.status_code == 200 and len(resposta.content) > TAMANHO_MINIMO_VALIDO:
                with open(destino, "wb") as arquivo:
                    arquivo.write(resposta.content)
                return True
        except requests.RequestException:
            pass
        tentativa = tentativa + 1
        time.sleep(1)
    # registra a falha indicando as coordenadas e continua sem interromper
    print("falha ao baixar tile z=" + str(z) + " x=" + str(x) + " y=" + str(y))
    return False


def listar_tiles(bbox, z):
    # converte o bbox em uma lista de coordenadas de tile a baixar
    lon_min, lat_min, lon_max, lat_max = bbox
    x_min, y_max = deg2tile(lat_min, lon_min, z)
    x_max, y_min = deg2tile(lat_max, lon_max, z)
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    coordenadas = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            coordenadas.append((z, x, y))
    return coordenadas


def coletar_regiao(nome, bbox, papel, z):
    # baixa todos os tiles de uma regiao para a pasta do seu papel no split
    pasta_saida = os.path.join(PASTA_BASE, papel, nome)
    os.makedirs(pasta_saida, exist_ok=True)
    print("coletando regiao", nome, "papel", papel, "bbox", bbox, "zoom", z)
    coordenadas = listar_tiles(bbox, z)
    sucesso = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        tarefas = []
        for z_tile, x, y in coordenadas:
            tarefas.append(executor.submit(baixar_tile, z_tile, x, y, pasta_saida))
        for tarefa in tarefas:
            if tarefa.result():
                sucesso = sucesso + 1
    print("regiao", nome, "tiles solicitados", len(coordenadas), "tiles salvos", sucesso)
    return len(coordenadas), sucesso


def main():
    # percorre todas as regioes definidas e baixa os tiles no zoom escolhido
    os.makedirs(PASTA_BASE, exist_ok=True)
    total_solicitado = 0
    total_salvo = 0
    for nome in REGIOES:
        bbox = REGIOES[nome]["bbox"]
        papel = REGIOES[nome]["papel"]
        solicitado, salvo = coletar_regiao(nome, bbox, papel, ZOOM)
        total_solicitado = total_solicitado + solicitado
        total_salvo = total_salvo + salvo
    print("coleta concluida. total solicitado", total_solicitado, "total salvo", total_salvo)
    print("Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community")


if __name__ == "__main__":
    main()
