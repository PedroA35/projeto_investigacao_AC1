# Gradient Boosting Classifier: Lidar com o Desequilíbrio de Classes

**UC:** Aprendizagem Computacional I — Universidade do Porto  


---

Este repositório contém a implementação "from scratch" do algoritmo **Gradient Boosting Classifier**, desenvolvida para a unidade curricular de Aprendizagem Computacional I (CC2008). O foco do projeto é analisar como este algoritmo se comporta em datasets desbalanceados e propor melhorias.

## 👤 Autores

Este projeto foi desenvolvido por:

| Membro | Curso e Número Mecanográfico | Perfil do LinkedIn |
| :--- | :--- | :--- |
| **Maria Marinho** | Bioinformática FCUP / 202403549 | (Link indisponível) |
| **Pedro Afonso** | Bioinformática FCUP / 202404125 | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pedro-afonso-282a43294/) |
| **Rafael Santos** | Bioinformática FCUP / 202405265 | (Link indisponível) |

---


## 🎯 1. Resumo do Projeto
O **Gradient Boosting Classifier** é um método de ensemble poderoso, mas que frequentemente sofre em datasets onde uma classe é muito mais frequente que a outra (Class Imbalance). Este trabalho explora:
1. A implementação base do algoritmo usando apenas **NumPy** e **SciPy**.
2. A avaliação experimental em datasets de benchmark.
3. A identificação de falhas nas métricas tradicionais (como Accuracy) em cenários de desequilíbrio.

---

### 🧠 2. Mecanismo de Gradient Boosting
Ao contrário de modelos que tentam resolver o problema de uma só vez, o nosso Gradient Boosting funciona por **estágios sucessivos**:

1.  **Previsão Inicial:** O modelo começa com um valor base: a probabilidade média da classe nos dados de treino.
2.  **Cálculo de Resíduos:** Em cada iteração, o algoritmo identifica onde errou na etapa anterior através da derivada da **LogLoss** (função de perda).
3.  **Treino de Weak Learners:** Uma nova **Árvore de Decisão** é treinada especificamente para prever estes erros, ou resíduos, e não o valor final.
4.  **Consolidação com Learning Rate:** Esta nova árvore é somada ao modelo acumulado, multiplicada por uma taxa de aprendizagem (*learning rate*). Este passo é fundamental, garantindo que o modelo não decore os dados (overfitting), o que permite uma evolução gradual e robusta.



---

## 📁 Estrutura de Ficheiros
* `gbm.py`: Contém a lógica do Gradient Boosting e funções de perda (LogLoss).
* `tree.py`: Implementação da Árvore de Decisão (weak learner) sem dependências externas.
* `final_assignment.py`: Script de automação que carrega os datasets da pasta `/data` e realiza:
    * **Pré-processamento:** Tratamento de valores omissos e encoding de variáveis categóricas.
    * **Validação:** Divisão aleatória (Hold-out 80/20) para garantir a integridade estatística dos testes.
* `data/`: Pasta onde devem ser colocados os ficheiros `.csv`.

---

## 🛠️ Como Executar
1. Colocar os datasets `.csv` na pasta `/data`.
2. Instalar dependências: `pip install numpy pandas scikit-learn scipy`
3. Correr o script principal: `python final_assignment.py`
