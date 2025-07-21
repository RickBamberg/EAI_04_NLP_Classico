# Fundamentos de Processamento de Linguagem Natural (NLP Clássico)

Este módulo reúne os principais conceitos e técnicas clássicas de NLP (Natural Language Processing), aplicados com Python e bibliotecas como scikit-learn, NLTK e Gensim. Cada notebook aborda uma etapa fundamental do pré-processamento e representação de texto, até a aplicação de modelos de classificação.

## Conteúdo dos Notebooks

### 📌 `pre_processamento_texto.ipynb`
Estudo sobre as etapas básicas de limpeza e normalização de texto:
- Remoção de pontuação e stopwords
- Tokenização e stemming
- Transformação em textos limpos para modelagem

---

### 📌 `representacao_bow_tfidf.ipynb`
Exploração dos principais métodos de vetorização de texto:
- Bag of Words (BoW)
- TF-IDF (Term Frequency - Inverse Document Frequency)
- Comparação entre as representações

---

### 📌 `bow_tfidf.ipynb`
Aplicação prática das representações vetoriais:
- Visualização de vetores gerados por BoW e TF-IDF
- Discussão sobre esparsidade e dimensionalidade

---

### 📌 `word_embeddings.ipynb`
Introdução aos **Word Embeddings**:
- Conceito de representações densas
- Geração de embeddings com Word2Vec (modelo Skip-gram)
- Visualização em 2D com PCA

---

### 📌 `pretrained_embeddings.ipynb`
Uso de **embeddings pré-treinados**:
- Carregamento do modelo Word2Vec Google News
- Extração e análise de semântica com similaridade de palavras
- Alternativas como FastText

---

### 📌 `analise_sentimentos.ipynb`
Projeto de **Análise de Sentimentos com Classificação de Texto**:
- Pré-processamento de frases com polaridade (positivo/negativo)
- Vetorização com BoW e TF-IDF
- Classificação com Naive Bayes e SVM
- Avaliação com métricas, testes com frases reais e análise de erros

---

## 🔍 Resumo Geral

Este módulo demonstra como transformar texto em dados utilizáveis por algoritmos de Machine Learning. O estudo percorre desde a limpeza do texto até classificações reais, combinando teoria e prática em NLP clássico.

---

