---

# **Análise de Feedback de Clientes: Sentimento e Intenção com NLP e Flask**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg) ![Flask](https://img.shields.io/badge/Flask-2.0%2B-black.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Visão Geral do Projeto

Este projeto é uma aplicação web completa que utiliza técnicas de Processamento de Linguagem Natural (PLN) para realizar uma análise dupla em feedbacks de clientes. A ferramenta não se limita a classificar o sentimento (Positivo ou Negativo), mas também identifica a **intenção** do texto, diferenciando uma opinião pura de uma sugestão de melhoria.

O objetivo é fornecer uma análise mais rica e acionável, permitindo que uma empresa não só entenda *como* os clientes se sentem, mas também capture *ideias e insights* valiosos para a evolução de seus produtos e serviços.

---

## ✨ Principais Funcionalidades

*   **Análise Dupla:** Processa qualquer texto em português e retorna duas predições independentes:
    1.  **Sentimento:** Classifica o texto como **Positivo** ou **Negativo**.
    2.  **Intenção:** Classifica se o texto contém uma **Sugestão** ou é uma **Opinião/Declaração**.
*   **Score de Confiança:** Exibe a probabilidade (confiança) de cada predição do modelo.
*   **Interface Web Interativa:** Desenvolvida com Flask, permite que qualquer usuário teste os modelos facilmente através de um navegador.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

*   **Linguagem:** Python 3.9
*   **Análise e Modelagem de Dados:**
    *   `Scikit-learn`: Para construção dos pipelines de Machine Learning (TF-IDF, Regressão Logística).
    *   `Pandas`: Para manipulação e preparação dos dados.
    *   `NLTK`: Para pré-processamento de texto (se utilizado).
    *   `Joblib`: Para serialização e salvamento dos modelos treinados.
*   **Desenvolvimento Web (Backend):** `Flask`
*   **Frontend:** HTML5, CSS3

---

## 🔬 Metodologia e Workflow

O projeto foi dividido em duas fases principais: Modelagem em um ambiente Jupyter Notebook e Deploy em uma aplicação web com Flask.

### 1. Coleta e Preparação dos Dados

Dois datasets distintos foram utilizados para treinar os dois modelos especializados:

*   **Para o Modelo de Sentimento:** Utilizou-se o dataset **B2W-Reviews01**, contendo mais de 130.000 avaliações de produtos. As notas foram mapeadas para as classes `Positivo` (notas 4 e 5) e `Negativo` (notas 1 e 2).
    Fonte: https://github.com/americanas-tech/b2w-reviews01/blob/main/B2W-Reviews01.csv
*   **Para o Modelo de Sugestão:** Diante da escassez de datasets públicos de sugestões, foi adotada uma abordagem de **Engenharia de Dados**:
    1.  Criação de um dataset sintético de alta qualidade com **1.506 exemplos de sugestões**, gerado com o auxílio de múltiplas IAs para garantir diversidade linguística.
    2.  Utilização de uma amostra do dataset B2W (filtrado para não conter sugestões) como exemplos da classe "Não-Sugestão".
    3.  Criação de um **dataset final balanceado** para o treinamento.

### 2. Modelagem e Treinamento

Foram construídos dois pipelines de classificação independentes, ambos utilizando a arquitetura `TF-IDF Vectorizer` + `Logistic Regression`.

*   **`sentiment_pipeline.pkl`:** Treinado para a tarefa de análise de sentimento.
*   **`suggestion_pipeline.pkl`:** Treinado para a tarefa de detecção de sugestão.

### 3. Resultados dos Modelos

Ambos os modelos atingiram performance de alta qualidade no conjunto de teste:

*   **Modelo de Sentimento:**
    *   Acurácia: **95%**
    *   F1-Score (Ponderado): **0.95**

*   **Modelo de Sugestão:**
    *   Acurácia: **98%**
    *   F1-Score (Ponderado): **0.98**

### 4. Deploy com Flask

Os dois pipelines salvos (`.pkl`) foram carregados em uma aplicação Flask, que expõe uma interface web para interação do usuário. A aplicação recebe o texto, passa-o pelos dois modelos e exibe os resultados de forma consolidada.

---

## 🚀 Como Executar o Projeto Localmente

**1. Pré-requisitos:**
*   Python 3.8 ou superior
*   `pip` (gerenciador de pacotes)

**2. Clone o Repositório:**
```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

**3. Crie e Ative um Ambiente Virtual (Recomendado):**
```bash
# Para Windows
python -m venv venv
.\venv\Scripts\activate

# Para macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**4. Instale as Dependências:**
```bash
pip install -r requirements.txt
```

**5. Execute a Aplicação Flask:**
```bash
python app.py
```

**6. Acesse no Navegador:**
Abra seu navegador e vá para `http://127.0.0.1:5000`

---

## 📂 Estrutura do Projeto

```
/projeto_analise_feedback/
├── app.py                  # Lógica da aplicação Flask
├── requirements.txt        # Dependências do Python
├── /models/
│   ├── sentiment_pipeline.pkl  # Modelo treinado para sentimento
│   └── suggestion_pipeline.pkl # Modelo treinado para sugestão
├── /notebooks/
│   └── Treinamento_Modelos.ipynb # Notebook com todo o processo de análise e treino
├── /static/
│   └── style.css           # Estilos da aplicação
└── /templates/
    ├── index.html          # Página inicial com o formulário
    └── resultado.html      # Página que exibe os resultados
```

---

## 🤔 Desafios e Aprendizados

*   **Escassez de Dados:** O principal desafio foi a ausência de um dataset público para a tarefa de detecção de sugestões. A solução encontrada (geração de dados sintéticos com IAs) foi um grande aprendizado em engenharia de dados e na resolução criativa de problemas.
*   **Análise de Textos Mistos:** Os testes revelaram que frases com sentimentos mistos (ex: "O produto é bom, mas deveria ter mais bateria") são um desafio para modelos baseados em Bag-of-Words, o que abre caminho para futuras melhorias.

### Futuras Melhorias

*   **Modelos de Deep Learning:** Substituir os modelos clássicos por arquiteturas baseadas em Transformers (como BERT) para capturar melhor o contexto e a relação entre as palavras.
*   **Análise de Aspectos:** Evoluir o projeto para um sistema de Análise de Sentimento Baseada em Aspectos (ABSA), identificando sobre qual *aspecto* do produto (bateria, câmera, atendimento) o cliente está opinando.

---

## 👤 Autor

*   **[Carlos Henrique Bamberg Marques]**
*   **LinkedIn:** [https://www.linkedin.com/in/carlos-henrique-bamberg-marques](https://www.linkedin.com/in/carlos-henrique-bamberg-marques/)
*   **GitHub:** [https://github.com/RickBamberg](https://github.com/RickBamberg/)
*   **Email:** [rick.bamberg@gmail.com](mailto:rick.bamberg@gmail.com)

---

## 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.