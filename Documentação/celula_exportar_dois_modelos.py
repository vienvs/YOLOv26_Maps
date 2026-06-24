# cole esta celula no fim do notebook (no colab), depois de treinar as duas rodadas
# e a comparacao yolo11n. ela gera os artefatos de cada modelo com nome proprio,
# que o app streamlit consome (seletor de modelo + aba de comparacao).
#
# variaveis usadas (ja existem no notebook): PASTA_BASE, PASTA_RUNS, caminho_data,
# melhor_rodada. o nome da rodada de comparacao deve ser 'comparacao_yolo11n'.

import os
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


def suavizar(valores, janela=5):
    # media movel simples para destacar a tendencia sem esconder os dados crus
    serie = list(valores)
    suave = []
    for i in range(len(serie)):
        inicio = max(0, i - janela + 1)
        soma = 0.0
        cont = 0
        for j in range(inicio, i + 1):
            soma = soma + serie[j]
            cont = cont + 1
        suave.append(soma / cont)
    return suave


def achar_coluna(df, alvo):
    # encontra a coluna cujo nome contem o texto alvo
    for nome in df.columns:
        if alvo in nome:
            return nome
    return None


def gerar_curvas(run_dir, sufixo):
    # le o results.csv da rodada e salva curva_map_<sufixo>.png e curva_loss_<sufixo>.png
    caminho_results = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(caminho_results):
        print('sem results.csv em', run_dir)
        return
    df = pd.read_csv(caminho_results)
    limpas = {}
    for nome in df.columns:
        limpas[nome] = nome.strip()
    df = df.rename(columns=limpas)

    epocas = df[achar_coluna(df, 'epoch')]
    col_map50 = achar_coluna(df, 'mAP50(')
    col_map5095 = achar_coluna(df, 'mAP50-95')

    plt.figure(figsize=(8, 5))
    plt.plot(epocas, df[col_map50], color='#1f77b4', linewidth=1, alpha=0.3)
    plt.plot(epocas, df[col_map5095], color='#ff7f0e', linewidth=1, alpha=0.3)
    plt.plot(epocas, suavizar(df[col_map50]), label='mAP@0.5', color='#1f77b4', linewidth=2)
    plt.plot(epocas, suavizar(df[col_map5095]), label='mAP@0.5:0.95', color='#ff7f0e', linewidth=2)
    plt.xlabel('epoca')
    plt.ylabel('mAP')
    plt.title('Evolucao da mAP por epoca (validacao) - ' + sufixo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    destino_map = os.path.join(PASTA_BASE, 'curva_map_' + sufixo + '.png')
    plt.savefig(destino_map, dpi=150, bbox_inches='tight')
    plt.show()

    col_tb = achar_coluna(df, 'train/box_loss')
    col_tc = achar_coluna(df, 'train/cls_loss')
    col_td = achar_coluna(df, 'train/dfl_loss')
    col_vb = achar_coluna(df, 'val/box_loss')
    col_vc = achar_coluna(df, 'val/cls_loss')
    col_vd = achar_coluna(df, 'val/dfl_loss')
    loss_treino = df[col_tb] + df[col_tc] + df[col_td]
    loss_val = df[col_vb] + df[col_vc] + df[col_vd]

    plt.figure(figsize=(8, 5))
    plt.plot(epocas, loss_treino, color='#1f77b4', linewidth=1, alpha=0.3)
    plt.plot(epocas, loss_val, color='#d62728', linewidth=1, alpha=0.3)
    plt.plot(epocas, suavizar(loss_treino), label='loss treino', color='#1f77b4', linewidth=2)
    plt.plot(epocas, suavizar(loss_val), label='loss validacao', color='#d62728', linewidth=2)
    plt.xlabel('epoca')
    plt.ylabel('loss total (box + cls + dfl)')
    plt.title('Loss de treino vs validacao por epoca - ' + sufixo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    destino_loss = os.path.join(PASTA_BASE, 'curva_loss_' + sufixo + '.png')
    plt.savefig(destino_loss, dpi=150, bbox_inches='tight')
    plt.show()


def exportar_modelo(run_dir, sufixo):
    # gera best_<sufixo>.pt, metricas_<sufixo>.json, confusion_matrix_<sufixo>.png e curvas
    pesos = os.path.join(run_dir, 'weights', 'best.pt')
    if not os.path.exists(pesos):
        print('sem best.pt em', run_dir)
        return
    print('exportando', sufixo, 'de', run_dir)

    # copia os pesos
    shutil.copy(pesos, os.path.join(PASTA_BASE, 'best_' + sufixo + '.pt'))

    # avalia no teste e salva as metricas
    modelo = YOLO(pesos)
    m = modelo.val(data=caminho_data, split='test', conf=0.25, iou=0.50)
    metricas = {
        'map50': float(m.box.map50),
        'map': float(m.box.map),
        'precisao': float(m.box.mp),
        'revocacao': float(m.box.mr),
    }
    with open(os.path.join(PASTA_BASE, 'metricas_' + sufixo + '.json'), 'w') as arq:
        json.dump(metricas, arq, indent=2)
    print(sufixo, metricas)

    # copia a matriz de confusao da avaliacao
    origem_matriz = os.path.join(str(m.save_dir), 'confusion_matrix.png')
    if os.path.exists(origem_matriz):
        shutil.copy(origem_matriz, os.path.join(PASTA_BASE, 'confusion_matrix_' + sufixo + '.png'))

    # gera as curvas a partir do historico de treino
    gerar_curvas(run_dir, sufixo)


# yolo26n: melhor rodada escolhida antes
exportar_modelo(os.path.join(PASTA_RUNS, melhor_rodada), 'yolo26n')

# yolo11n: rodada de comparacao
exportar_modelo(os.path.join(PASTA_RUNS, 'comparacao_yolo11n'), 'yolo11n')

print('pronto. baixe do drive e coloque na pasta app/ do repositorio:')
print('best_yolo26n.pt, best_yolo11n.pt, metricas_yolo26n.json, metricas_yolo11n.json,')
print('confusion_matrix_yolo26n.png, confusion_matrix_yolo11n.png,')
print('curva_map_yolo26n.png, curva_loss_yolo26n.png, curva_map_yolo11n.png, curva_loss_yolo11n.png')
