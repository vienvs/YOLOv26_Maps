**Estruturação do Dataset:**



* **Split:**



Primeiramente definimos quais bairros utilizar para Treino/Evaluate e qual bairro para Teste, decidimos:

&#x20;

Treino e Validação (Mesmo dataset, pois validação monitora overfitting, e deveria estar no mesmo split)

&#x20;- Itaim Bibi (Faria Lima) (-46.694, -23.592, -46.672, -23.572)

&#x20;- Vila Olímpia (-46.695, -23.604, -46.676, -23.590)

&#x20;- Brooklyn/ Berrini (-46.703, -23.624, -46.683, -23.606)

&#x20;- Pinheiros (-46.710, -23.575, -46.696, -23.560)

&#x20;- Morumbi (-46.730, -23.612, -46.708, -23.590)

&#x20;- Chucri Zaidan (-46.726, -23.628, -46.706, -23.614)

&#x20;- Santo Amaro (-46.716, -23.648, -46.700, -23.630)

&#x20;- Moema (-46.670, -23.605, -46.655, -23.590)

&#x20;- Campo Belo (-46.676, -23.625, -46.661, -23.610)

&#x20;- Centro (-46.645, -23.550, -46.630, -23.536)

&#x20;- Higienópolis (-46.665, -23.548, -46.650, -23.535)

&#x20;- Pacaembu (-46.685, -23.545, -46.665, -23.530)

&#x20;- Faria Lima (FIX) (-46.695, -23.572, -46.678, -23.560)



Teste:

&#x20;- Avenida Paulista (Tem características visuais ligeiramente diferentes dos outros três (edifícios mais antigos, hotéis e etc), o que vai estressar bem o modelo)

&#x20;(-46.660, -23.572, -46.640, -23.555)



* **Imagens:**



Decidi salvar no dataset de treino e validação tanto imagens com helipontos quanto sem, para garantir que o modelo também aprenda o que não é um heliponto,

pretendo fazer um dataset com 250 imagens, salvando todas as com heliponto dos bairros e acrescentando as que não tem para preencher a satisfaçã



* **Curadoria via Roboflow:**



Com todas as imagens dos 4 bairros, separar explicitamente e tratar em outro projeto as imagens da Paulista, fazer o Bounding Box e o Label de todas as imagens.



Para Treino/Validação:

&#x20;- Salvar TODAS as com Heliponto e salvar algumas que não tem (Para fechar 250 imagens)



Para Teste:

&#x20;- Salvas todas as imagens, tendo heliponto ou não



* **Tratamento:**



Como vou usar Helipontos, tenho que usar z = 19 para zoom, fazendo com que eu tenha que gerar 4x mais imagens, mas para facilitar a audição e aumentar qualidade das

imagens (permitindo melhor visualização para o modelo) vou juntar 4 tiles e formar uma imagem 512x512, assim como no notebook Projeto\_P2\_Mosaico\_Perdizes\_HIRES.ipynb,

porque além desses benefícios, o upscale para 640x640 é muito mais sutil e perde menos qualidade.


Peguei algumas imagens do Google Earth para complementar e chegar a 200 alvos, garantindo ângulos diferentes e fotos diferentes de pontos das mesmas regiões.



Dataset de Treino/Val tem 160 imagens, sendo cerca de 30 com falsos positivos para o modelo saber o que não é



Dataset de Teste tem 195 imagens



Como varri a Paulista inteira, vai ter mais imagens, porém com várias sem nada, o que pode afetar o mAP do teste com variância maior



Ao todo, reuni 201 incidências de Helipontos, em cerca de 150 imagens.



Cerca de 2.4k imagens sem incluir no dataset, com curadoria pesada.



Para o tratamento final, adicionei a etapa no Roboflow para pré processamento, dando o output já em 640x640, além do Augmentation para rotação 90° e adição de Brilho de +- 20%



Ao final, foram 419 imagens para Treino, 20 para validação e 195 para teste



Código para o Colab: 

!pip install roboflow



from roboflow import Roboflow

rf = Roboflow(api\_key="l11c8wR2xo4QODKQZCJM")

project = rf.workspace("yolo-maps-pucsp-vie").project("yolov26-maps-vie")

version = project.version(1)

dataset = version.download("yolo26")

&#x20;               

