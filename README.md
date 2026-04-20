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

### ⚖️ Escala de Dificuldade: Imbalance Ratio (IR)
O **IR** é o indicador fundamental da dificuldade do problema, medindo a proporção entre a classe minoritária e a majoritária. Classificámos os datasets segundo a seguinte escala de severidade:

| Categoria | Intervalo IR | Exemplo no Projeto | Impacto Esperado |
| :--- | :--- | :--- | :--- |
| **Extremo** | IR < 0.05 | `yeast_ml8`, `oil_spill` | O modelo tende a ignorar a classe rara, resultando em F1-Score zero. |
| **Moderado** | 0.05 < IR < 0.15 | `hypothyroid`, `sick` | O modelo identifica alguns padrões, mas sofre com elevados falsos negativos. |
| **Suave** | IR > 0.15 | `backache`, `chlamydia` | O desequilíbrio é menos punitivo e os resultados de F1 são mais estáveis. |



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
