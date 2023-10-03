from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A classificação de subtipos tumorais de gliomas a partir de marcadores moleculares é uma metodologia que tem crescido nos últimos anos. Isso já foi feito utilizando-se do perfil de metilação de DNA de pacientes através do algoritmo de uma Random Forest, por um estudo de 2016. No entanto, dados de expressão gênica são mais disponíveis e viáveis de sequenciamento no contexto clínico em comparação à metilação de DNA, apresentando um maior potencial de aplicação na rotina de hospitais. O impasse no uso de dados de sequenciamento de expressão gênica (RNA-seq) se dá pelo alto número de atributos (genes) inicialmente. Propõe-se realizar a fase de seleção de atributos via Algoritmos Genéticos (AGs), a fim de propor o conjunto de genes ótimo para o desenvolvimento de um novo classificador de subtipos de gliomas com maior potencial de aplicabilidade no contexto clínico.',
    author='Isabela Erthal e Sergio Baldo',
    license='MIT',
)
