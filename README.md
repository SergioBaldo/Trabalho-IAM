Trabalho da disciplina de Introdução ao Aprendizado de Máquina "Seleção de Atributos via Algoritmos Genéticos para a classificação de subtipos de glioma a partir de dados de expressão gênica"
==============================

A classificação de subtipos tumorais de gliomas a partir de marcadores moleculares é uma metodologia que tem crescido nos últimos anos. Isso já foi feito utilizando-se do perfil de metilação de DNA de pacientes através do algoritmo de uma Random Forest, por um estudo de 2016. No entanto, dados de expressão gênica são mais disponíveis e viáveis de sequenciamento no contexto clínico em comparação à metilação de DNA, apresentando um maior potencial de aplicação na rotina de hospitais. O impasse no uso de dados de sequenciamento de expressão gênica (RNA-seq) se dá pelo alto número de atributos (genes) inicialmente. Propõe-se realizar a fase de seleção de atributos via Algoritmos Genéticos (AGs), a fim de propor o conjunto de genes ótimo para o desenvolvimento de um novo classificador de subtipos de gliomas com maior potencial de aplicabilidade no contexto clínico.

Organização do Projeto
------------

    ├── LICENSE            <- Termos de licença que regem o uso do software.
    ├── README.md          <- Documentação principal que fornece informações sobre o projeto, incluindo uma descrição geral e instruções de uso.
    ├── data               <- Esta pasta contém subpastas relacionadas ao gerenciamento de dados.
    │   ├── interim        <- Local onde dados temporários ou em processo são armazenados.
    │   ├── processed      <- Contém dados após processamento.
    │   └── raw            <- Dados brutos sem qualquer processamento.
    |
    ├── references         <- Referências relevantes para o projeto.
    │
    ├── reports            <- Pasta destinada a relatórios e resultados.
    │   └── results        <- Armazena os resultados das execuções do Algoritmo Genético
    │
    ├── requirements.txt   <- Lista das dependências do projeto que podem ser instaladas usando um gerenciador de pacotes.
    │
    ├── .env               <- Arquivo contendo as variáveis de ambiente.
    │
    ├── src                <- Esta pasta contém o código-fonte do projeto.
    │   ├── __init__.py    <- Arquivo que indica que o diretório deve ser considerado um pacote Python.
    │   │
    │   ├── data                    <- Subpasta relacionada ao manuseio de dados.
    │   │   └── make_dataset.py     <- Script para criar os conjuntos de dados em treino e teste.
    │   │
    │   │
    │   ├── genetic_algorithm       <- Contém scripts relacionados ao algoritmo genético.
    │   │   └── ga_functions.py     <- Funções principais do algoritmo genético.
    │   │   └── genetic_search.py   <- Implementação do algoritmo de busca genética.
    │   │   └── save_ga_results.py  <- Script para salvar resultados do algoritmo genético.
    │   │
    │   ├── util                                        <- Contém scripts utilitários.
    │   │   └── feature_selection_and_evaluation.py     <- Script para a avaliação da seleção de atributos.
    |   |   └── utils_ga.py                             <- Funções utilitárias para o algoritmo genético.
    │   │ 
    │   ├── wrapper             <- Contém scripts relacionados a "wrapper methods".
    │       └── wrapper.py      <- Implementação de métodos de "wrapper".
    |   
    │
    └── main.py     <- O arquivo principal para executar o projeto.

--------

## Criando um Ambiente Virtual Para Executar o projeto

Como criar um ambiente virtual usando tanto o Conda quanto o método `venv` do Python.

### Com Conda

1. Abra o seu terminal ou prompt de comando.

2. Execute o seguinte comando para criar um ambiente Conda. Substitua `meuambiente` pelo nome de ambiente desejado:

`conda create --name meuambiente`

3. Para ativar o ambiente Conda, use o seguinte comando:

`conda activate meuambiente`

4. Para instalar os pacotes Python usando `conda` ou `pip` dentro do ambiente basta usar o seguinte comando: 

`pip install -r requirements.txt`

### Com Python (venv)

1. Abra o seu terminal ou prompt de comando.

2. Navegue até o diretório onde deseja criar o ambiente virtual.

3. Execute o seguinte comando para criar um ambiente virtual usando o `venv` integrado do Python. Substitua `meuambiente` pelo nome de ambiente desejado:

`python -m venv meuambiente`

4. Para ativar o ambiente virtual, forneça instruções com base no sistema operacional dos usuários (Windows, macOS ou Linux).
    #### Windows
    `meuambiente\Scripts\activate`

    #### macOS ou Linux:
    `source meuambiente/bin/activate`

5. Para instalar os pacotes Python usando `pip` dentro do ambiente, basta usar o seguinte comando: 

`pip install -r requirements.txt`


## Para Executar o projeto

Para executar o projeto basta utilizar o seguinte comando:

`python main.py`
