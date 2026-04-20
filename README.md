# Gradient Boosting Classifier: Lidar com o Desequilíbrio de Classes

**UC:** Aprendizagem Computacional I — Universidade do Porto  


---

Este repositório contém a implementação "from scratch" do algoritmo **Gradient Boosting Classifier**, desenvolvida para a unidade curricular de Aprendizagem Computacional I (CC2008). O foco do projeto incide na análise do comportamento dete algoritmo em conjuntos de dados desequilibrados, bem como na proposta de melhorias estruturais.


## 👤 Autores

Este projeto foi desenvolvido por:

| Membro | Curso e Número Mecanográfico | Perfil do LinkedIn | Perfil do GitHub |
| :--- | :--- | :--- | :--- |
| **Maria Marinho** | Bioinformática FCUP / 202403549 | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/maria-duarte-marinho-a7b655348/) | [![GitHub](https://img.shields.io/badge/GitHub-Perfil-black?logo=github)](https://github.com/madu615) |
| **Pedro Afonso** | Bioinformática FCUP / 202404125 | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pedro-afonso-282a43294/) | [![GitHub](https://img.shields.io/badge/GitHub-Perfil-black?logo=github)](https://github.com/PedroA35) |
| **Rafael Santos** | Bioinformática FCUP / 202405265 | (Link indisponível) | [![GitHub](https://img.shields.io/badge/GitHub-Perfil-black?logo=github)](https://github.com/Rafael00Santos) |


---


## 🎯 1. Resumo do Projeto

O **Gradient Boosting Classifier** é um método de *ensemble* (combinação de múltiplos modelos) robusto. No entanto, apresenta frequentemente limitações em conjuntos de dados onde uma classe é significativamente mais frequente do que a outra, problema conhecido como *class imbalance*. Este trabalho explora:

1. A implementação base do algoritmo, utilizando exclusivamente **NumPy** e **SciPy**.
2. A avaliação experimental em conjuntos de dados de referência (*benchmark*).
3. A identificação de limitações das métricas tradicionais, como a *accuracy*, em cenários de desequilíbrio acentuado.

---


## 🧠 2. Mecanismo de Gradient Boosting
Ao contrário de modelos que tentam resolver o problema de uma só vez, o **Gradient Boosting** funciona por **estágios sucessivos**:

1.  **Previsão inicial:** O modelo começa com um valor base: a probabilidade média da classe nos dados de treino.
2.  **Cálculo de resíduos:** Em cada iteração, o algoritmo identifica os erros da etapa anterior, através da derivada da **log loss** (função de perda).
3.  **Treino de modelos fracos (*weak learners*):** Uma nova **árvore de decisão** é treinada especificamente para prever estes erros (resíduos) e não o valor final.
4.  **Atualização do modelo com taxa de aprendizagem:** Esta nova árvore é adicionada ao modelo acumulado, após ser multiplicada por uma taxa de aprendizagem (*learning rate*). Este passo é fundamental, para evitar sobreajuste (overfitting), permitindo uma evolução gradual e robusta.


---


## ⚖️ 3. Escala de Dificuldade: *Imbalance Ratio* (IR)
O **IR** é um indicador fundamental da dificuldade do problema, que mede a proporção entre as classes minoritária e maioritária. Os conjuntos de dados foram calssificados, segundo a seguinte escala de severidade:

| Categoria | Intervalo IR | Exemplo no Projeto | Impacto Esperado |
| :--- | :--- | :--- | :--- |
| **Extremo** | IR < 0.05 | `yeast_ml8`, `oil_spill` | O modelo tende a ignorar a classe rara, resultando num valor de F1-score muito baixo ou nulo. |
| **Moderado** | 0.05 < IR < 0.15 | `hypothyroid`, `sick` | O modelo identifica alguns padrões, mas apresenta um numero elevado de falsos negativos. |
| **Suave** | IR > 0.15 | `backache`, `chlamydia` | O desequilíbrio é menos crítico e os resultados de F1-score são mais estáveis. |



### ⚖️ 3.1. Análise Comparativa: Vantagens e Limitações

A utilização do Gradient Boosting apresenta um conjunto de características fundamentais que motivaram a sua escolha:

* **Pontos fortes:**

   | Característica | Explicação |
   | :--- | :--- |
   | **Elevado poder preditivo** | Frequentemente apresenta resultados superiores em dados estruturados, devido à sua natureza aditiva. |
   | **Flexibilidade** | Permite a otimização de diferentes funções de perda (como a *log loss* utilizada neste projeto). |
   | **Capacidade de capturar relações não lineares** | Capta relações complexas entre variáveis que modelos lineares não conseguem representar. |


* **Limitações:**

   | Característica | Explicação |
   | :--- | :--- |
   | **Sensibilidade ao ruído** | Pode sofrer de *overfitting* se o número de árvores for excessivo ou a taxa de aprendizagem demasiado elevada. |
   | **Sensibilidade ao desequilíbrio** | Na sua forma padrão (Fase 1), foca-se na minimização do erro global, penalizando a classe minoritária. |
   | **Custo computacional elevado** | O treino é sequencial, tornando-o mais lento do que modelos que permitem execução em paralelo, como o *Random Forest*. |


---


## 🛠️ 4. Pipeline de Tratamento de Dados

Para garantir que o modelo de Gradient Boosting processa corretamente os conjuntos de dados de benchmark, o script `final_assignment.py` executa um fluxo de pré-processamento rigoroso:

- **Codificação de variáveis categóricas (Label Encoding):** Conversão automática de colunas de texto em valores numéricos, assegurando compatibilidade com operações numéricas do NumPy.
- **Tratamento de valores em falta:** Preenchimento de valores em falta com base na moda (valor mais frequente), para variáveis categóricas, e na média, para variáveis numéricas. Esta abordagem evita a remoção de observações (*dropna*), o que seria crítico em classes minoritárias com poucas amostras.
- **Validação hold-out:** Separação de 20% dos dados para teste. As métricas apresentadas são calculadas exclusivamente neste conjunto “invisível”, garantindo a integridade da avaliação.


---


## 📁 5. Estrutura de Ficheiros
* `gbm.py`: contém a lógica do Gradient Boosting e a função de perda (*log loss*).
* `tree.py`: implementação da árvore de decisão (*weak learner*) sem dependências externas.
* `final_assignment.py`: script de automação que carrega os conjuntos de dados da pasta `/data` e realiza:
    * **Pré-processamento:** tratamento de valores omissos e codificação de variáveis categóricas.
    * **Validação:** divisão aleatória (hold-out 80/20) para garantir a integridade estatística da avaliação.
* `data/`: pasta onde devem ser colocados os ficheiros `.csv`.


---


## 🛠️ 6. Como Executar

1. Colocar os ficheiros `.csv` na pasta `/data`.
2. Instalar as dependências:
   ```bash
   pip install numpy pandas scikit-learn scipy
3. Executar o script principal:
   ```bash
   python final_assignment.py
---


## 🏁 7. Conclusões da Fase 1

Os resultados experimentais validam a hipótese central da investigação: o Gradient Boosting padrão, ao minimizar a função de perda global (*log loss*), tende a favorecer a classe maioritária.

- ***Accuracy* enganadora:** Em conjuntos de dados como `oil_spill`, foram obtidos *accuracy values* superiores a 95%, apesar de um F1-score nulo. Isto evidencia que o modelo pode otimizar a métrica global sem capturar adequadamente a classe minoritária, privilegiando a classe dominante.
- **Fundamentação para melhorias:** Estes resultados justificam a necessidade de evoluir para abordagens como funções de custo ponderadas ou ajuste de limiares, que serão exploradas em fases futuras para melhorar a deteção da classe minoritária.
