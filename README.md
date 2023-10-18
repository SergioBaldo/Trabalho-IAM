Trabalho da disciplina de Introdução ao Aprendizado de Máquina "Seleção de Atributos via Algoritmos Genéticos para a classificação de subtipos de glioma a partir de dados de expressão gênica"
==============================

A classificação de subtipos tumorais de gliomas a partir de marcadores moleculares é uma metodologia que tem crescido nos últimos anos. Isso já foi feito utilizando-se do perfil de metilação de DNA de pacientes através do algoritmo de uma Random Forest, por um estudo de 2016. No entanto, dados de expressão gênica são mais disponíveis e viáveis de sequenciamento no contexto clínico em comparação à metilação de DNA, apresentando um maior potencial de aplicação na rotina de hospitais. O impasse no uso de dados de sequenciamento de expressão gênica (RNA-seq) se dá pelo alto número de atributos (genes) inicialmente. Propõe-se realizar a fase de seleção de atributos via Algoritmos Genéticos (AGs), a fim de propor o conjunto de genes ótimo para o desenvolvimento de um novo classificador de subtipos de gliomas com maior potencial de aplicabilidade no contexto clínico.

Organização do Projeto Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── results        <- Generated graphics and figures to be used in reporting         
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── .env               <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── genetic_algorithm       <- Scripts to turn raw data into features for modeling
    │   │   └── ga_functions.py
    │   │   └── genetic_search.py
    │   │   └── save_ga_results.py
    │   │   └── utils.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── wrapper       <- Scripts to turn raw data into features for modeling
    │   │   └── wrapper.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── main.py            <- 


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


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
