**Estruturação do Dataset:**



* **Split:** 



Primeiramente definimos quais bairros utilizar para Treino/Evaluate e qual bairro para Teste, decidimos:

&#x20;

Treino e Validação (Mesmo dataset, pois validação monitora overfitting, e deveria estar no mesmo split)

&#x20;- Itaim Bibi (Faria Lima) (-46.694, -23.592, -46.672, -23.572)

&#x20;- Vila Olímpia (-46.695, -23.604, -46.676, -23.590)

&#x20;- Brooklyn/ Berrini (-46.703, -23.624, -46.683, -23.606)



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





