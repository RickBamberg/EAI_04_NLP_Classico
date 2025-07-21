# 📦 Modelos de Classificação de Sentimentos

Este módulo apresenta diferentes abordagens de **modelos clássicos de aprendizado de máquina** aplicados à tarefa de **classificação de sentimentos em textos curtos**. Utilizamos um dataset rotulado com frases positivas e negativas, abordando o fluxo completo desde o pré-processamento até a avaliação e comparação dos modelos.

## 📑 Notebooks

### 1. `naive_bayes_sentimentos.ipynb`
Implementa um classificador de sentimentos utilizando o algoritmo **Naive Bayes Multinomial** com vetores de características baseados em **TF-IDF**.

- Pré-processamento de texto com remoção de stopwords e stemming.
- Conversão dos dados textuais para vetores TF-IDF.
- Treinamento e avaliação com métricas como acurácia, precisão, recall e F1-score.
- Matriz de confusão e análise de erros.
- Teste com frases manuais.

### 2. `classificacao_texto_svm.ipynb`
Aplica o algoritmo de **Máquinas de Vetores de Suporte (SVM)** para a mesma tarefa de classificação.

- Reutiliza o pipeline de TF-IDF e pré-processamento.
- SVM com kernel linear, usando `LinearSVC`.
- Avaliação completa e comparação de desempenho com o Naive Bayes.
- Inclui validação cruzada e testes com frases reais.

### 3. `comparativo_tfidf_vs_embeddings.ipynb`
Compara duas representações vetoriais distintas para textos:

- **TF-IDF:** método clássico baseado na frequência dos termos.
- **Word2Vec:** embeddings pré-treinados (Google News) como média vetorial das palavras.
  
Ambos os métodos são usados com um modelo **Logistic Regression** para prever sentimentos.

- Avaliação de desempenho com métricas e visualização em gráfico de barras.
- Discussão das vantagens e limitações de cada abordagem.

## 🎯 Objetivos do Módulo

- Explorar **modelos de classificação supervisionada** para problemas de NLP.
- Comparar o impacto de diferentes **representações vetoriais de texto**.
- Entender as **vantagens e limitações** de modelos clássicos aplicados ao sentimento.
- Consolidar habilidades de pré-processamento, modelagem e análise de resultados.

## ✅ Requisitos

- Python 3.8+
- scikit-learn
- pandas
- seaborn
- nltk
- gensim (para Word2Vec)
- matplotlib

Instale com:

```bash
pip install -r requirements.txt
