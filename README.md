# Gradient Boosting Classifier: Lidar com o Desequilíbrio de Classes

**UC:** Aprendizagem Computacional I — Universidade do Porto  


---

Este repositório contém a implementação "from scratch" do algoritmo **Gradient Boosting Classifier**, desenvolvida para a unidade curricular de Aprendizagem Computacional I (CC2008). O foco do projeto é analisar como este algoritmo se comporta em datasets desbalanceados e propor melhorias.

## 👤 Autores

Este projeto foi desenvolvido por:

| Membro | Curso e Número Mecanográfico | Perfil do LinkedIn |
| :--- | :--- | :--- |
| **Afonso Marcos** | Bioinformática FCUP / 202404088 | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/afonsomarcos20/) |
| **Pedro Afonso** | Bioinformática FCUP / 202404125 | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pedro-afonso-282a43294/) |
| **Artur Anselmo** | Bioinformática FCUP / 202403965 | (Link indisponível) |
| **Marcos Torres** | Bioinformática FCUP / 202409299 | (Link indisponível) |


* **Maria Marinho** - 202403549
* **Pedro Afonso** - 202404125
* **Rafael Santos** - 202405265

---


## 🎯 Resumo do Projeto
O Gradient Boosting é um método de ensemble poderoso, mas que frequentemente sofre em datasets onde uma classe é muito mais frequente que a outra (Class Imbalance). Este trabalho explora:
1. A implementação base do algoritmo usando apenas **NumPy** e **SciPy**.
2. A avaliação experimental em datasets de benchmark.
3. A identificação de falhas nas métricas tradicionais (como Accuracy) em cenários de desequilíbrio.

## 📁 Estrutura de Ficheiros
* `gbm.py`: Contém a lógica do Gradient Boosting e funções de perda (LogLoss).
* `tree.py`: Implementação da Árvore de Decisão (weak learner) sem dependências externas.
* `final_assignment.py`: Script de automação que carrega os datasets da pasta `/data`, realiza o pré-processamento e executa os testes.
* `data/`: Pasta onde devem ser colocados os ficheiros `.csv`.

## 🛠️ Instalação e Execução
1. Certifica-te de que tens o Python 3.x instalado.
2. Instala as dependências necessárias para manipulação de dados:
   ```bash
   pip install numpy pandas scikit-learn scipy
