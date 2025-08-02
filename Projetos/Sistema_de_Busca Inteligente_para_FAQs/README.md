# **Sistema de Busca Inteligente para FAQs**

![Python](https://img.shields.io/badge/Python-3.9-blue.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-orange.svg) ![Flask](https://img.shields.io/badge/Flask-black.svg) ![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-2.2.2-red.svg)

## 📖 Visão Geral do Projeto

Este projeto é uma aplicação web que implementa um sistema de **busca semântica** para uma base de conhecimento de Perguntas Frequentes (FAQs). Diferente de uma busca tradicional por palavras-chave, esta ferramenta entende o **significado** e o **contexto** da pergunta do usuário para encontrar as respostas mais relevantes, mesmo que as palavras usadas não sejam exatamente as mesmas.

O objetivo é simular uma solução de autoatendimento inteligente, capaz de reduzir a carga sobre equipes de suporte ao fornecer respostas precisas e instantâneas para as dúvidas mais comuns dos usuários.

---

## ✨ Principais Funcionalidades

*   **Busca por Significado:** Utiliza modelos de linguagem pré-treinados (`Sentence Transformers`) para comparar a semântica da pergunta do usuário com a base de FAQs.
*   **Ranking de Relevância:** Retorna as **3 respostas mais prováveis**, ordenadas por um score de similaridade (de 0% a 100%).
*   **Limite de Confiança:** Apenas exibe resultados que atingem um limiar mínimo de relevância (50%), evitando respostas sem sentido para perguntas fora do escopo do FAQ.
*   **Interface Web Interativa:** Construída com Flask, oferece uma experiência de usuário simples e direta para testar a busca.

---

## 🔬 Metodologia e Workflow

O projeto foi desenvolvido em duas etapas principais: preparação dos dados e construção da aplicação web.

### 1. Coleta e Preparação dos Dados

*   **Fonte de Dados:** Foi utilizado um dataset público contendo **1.172 FAQs do Banco Central do Brasil**, extraído em formato JSON a partir do Portal de Dados Abertos do BCB.
    *   *Link da fonte original:* `https://www.bcb.gov.br/api/servico/faq/faqperguntas`
*   **Limpeza e Pré-processamento:** O campo de "resposta" continha dados brutos em HTML. Foi desenvolvido um pipeline de limpeza utilizando a biblioteca **BeautifulSoup** para extrair o texto puro, removendo todas as tags e garantindo um conteúdo limpo para o usuário final.

### 2. Vetorização e Busca Semântica

*   **Modelo de Embeddings:** Foi utilizado o modelo `distiluse-base-multilingual-cased-v1` da biblioteca **Sentence Transformers**. Este modelo é treinado para entender o significado de sentenças em múltiplos idiomas e convertê-las em vetores numéricos de 512 dimensões.
*   **Criação da Base de Conhecimento:** Cada uma das 1.172 perguntas do FAQ foi convertida em um vetor, e o conjunto desses vetores foi salvo como uma matriz NumPy (`embeddings_faq.npy`).
*   **Mecanismo de Busca:** A lógica de busca calcula a **similaridade de cosseno** entre o vetor da pergunta do usuário e todos os vetores da base de conhecimento. Os resultados com maior similaridade são então retornados.

### 3. Deploy com Flask

*   Os artefatos gerados (a matriz de embeddings e um arquivo `.pkl` com os textos originais) foram integrados a uma aplicação **Flask** que serve como interface para o sistema.

---

## 🛠️ Tecnologias Utilizadas

*   **Linguagem:** Python
*   **Análise de Dados:** Pandas, NumPy
*   **NLP / IA:** Sentence-Transformers, Scikit-learn
*   **Limpeza de Dados:** BeautifulSoup4
*   **Desenvolvimento Web:** Flask

---

## 🚀 Como Executar o Projeto Localmente

**1. Pré-requisitos:**
*   Ter o [Anaconda](https://www.anaconda.com/products/distribution) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instalado.

**2. Clone o Repositório:**

git clone https://github.com/RickBamberg/Sistema_de_Busca-Inteligente_para_FAQs.git
cd Sistema_de_Busca-Inteligente_para_FAQs

**3. Crie e Ative o Ambiente Virtual:**

* Cria o ambiente a partir do Python 3.9 (ou outra versão que preferir)
conda create --name faq_env python=3.9

* Ativa o ambiente recém-criado
conda activate faq_env

**4. Instale as Dependências:**

pip install -r requirements.txt

**5. Execute a Aplicação Flask:**

python app.py

**6. Acesse no Navegador:**

Abra seu navegador e vá para http://127.0.0.1:5000

---

## 📂 Estrutura do Projeto

/Sistema de Busca Inteligente para FAQs/  
├── app.py                  # Lógica da aplicação Flask  
├── requirements.txt        # Dependências do Python  
├── /models/  
│   ├── dados_faq.pkl       # Perguntas e respostas originais  
│   └── embeddings_faq.npy  # Matriz de embeddings  
├── /notebooks/  
│   └── FAQ_Semantic_Search.ipynb # Notebook com o processo de ETL e vetorização  
├── /static/  
│   └── style.css           # Estilos da aplicação  
└── /templates/  
    ├── index.html          # Página inicial com o formulário  
    └── resultado.html      # Página que exibe os resultados  

---

## 👤 Autor
**Carlos Henrique Bamberg Marques**
- GitHub: [https://github.com/RickBamberg](https://github.com/RickBamberg/)
- LinkedIn: [https://www.linkedin.com/in/carlos-henrique-bamberg-marques](https://www.linkedin.com/in/carlos-henrique-bamberg-marques/)
- Email: [rick.bamberg@gmail.com](mailto:rick.bamberg@gmail.com)

---

## 📜 Licença

Este projeto está sob a licença MIT.

---
