# AGENT_CONTEXT.md - Análise de Feedback (Dupla Classificação NLP)

> **Propósito**: Contexto técnico completo do sistema de análise de feedback  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Projeto real com dupla classificação e deployment Flask

## RESUMO EXECUTIVO

**Objetivo**: Análise automática de feedbacks com 2 classificações independentes  
**Modelos**: 2 pipelines TF-IDF + Logistic Regression  
**Datasets**: B2W-Reviews01 (129k) + Sugestões IA (1.5k)  
**Performance**: Sentimento 95%, Sugestão 98%  
**Deployment**: Flask web app  
**Diferencial**: Dois modelos especializados > um modelo genérico

---

## PROBLEMA - DUPLA CLASSIFICAÇÃO

### Desafio

Feedbacks de clientes contêm **múltiplas informações**:
1. **Sentimento**: Opinião (positiva/negativa)
2. **Tipo**: Sugestão de melhoria ou apenas opinião

**Exemplos**:
```
"Adorei o produto! Sugiro adicionar mais cores."
→ Sentimento: Positivo
→ Sugestão:   Sim

"Produto horrível, não recomendo."
→ Sentimento: Negativo
→ Sugestão:   Não
```

### Por Que 2 Modelos?

**Abordagem 1 (Não usada)**: Classificação multi-label
```python
# Problema: 4 classes possíveis
classes = [
    'Positivo + Sugestão',
    'Positivo + Não-Sugestão',
    'Negativo + Sugestão',
    'Negativo + Não-Sugestão'
]
# Complexo, requer mais dados
```

**Abordagem 2 (Usada)**: 2 modelos binários independentes ✅
```python
# Modelo 1: Sentimento (Positivo/Negativo)
# Modelo 2: Sugestão (Sim/Não)
# Simples, modular, melhor performance
```

---

## DATASET 1: B2W-REVIEWS01

### Fonte
- **URL**: https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets
- **Tamanho**: 129.098 avaliações
- **Idioma**: Português brasileiro
- **Domínio**: E-commerce (produtos variados)

### Estrutura
```python
df_b2w.columns
['review_id', 'product_id', 'reviewer_id', 'reviewer_name',
 'review_title', 'review_text', 'review_date', 'overall_rating',
 'recommend_to_a_friend', 'review_state']
```

### Campos Usados
```python
# review_text: Texto da avaliação
"Produto excelente, entrega rápida!"

# overall_rating: Nota de 1 a 5
overall_rating = 5
```

### Limpeza
```python
# Remover NaN
df_b2w.dropna(subset=['review_text'], inplace=True)

# Resultado: 129.098 linhas válidas
```

---

## DATASET 2: SUGESTOES.TXT

### Fonte
- **Geração**: Múltiplas IAs (GPT, Claude, etc.)
- **Tamanho**: 1.506 linhas
- **Formato**: Texto puro (uma sugestão por linha)

### Exemplos
```
Sugiro que implementem um chat de suporte 24/7
Seria interessante adicionar filtro de busca avançado
Poderiam melhorar o tempo de resposta do site
Recomendo que ofereçam mais opções de pagamento
```

### Características
- Variação de formalidade (formal ↔ informal)
- Diferentes domínios (produto, serviço, UX)
- Estruturas variadas (imperativo, condicional)

---

## PIPELINE DE DADOS

### Modelo 1: Sentimento

#### Mapeamento de Ratings
```python
def map_sentiment(rating):
    if rating <= 2:
        return 0  # Negativo
    elif rating >= 4:
        return 1  # Positivo
    else:
        return None  # Neutro (ignorado)

df['sentiment'] = df['overall_rating'].apply(map_sentiment)
df.dropna(subset=['sentiment'], inplace=True)
```

**Distribuição Final**:
```
Positivo (1): 79.316 (70%)
Negativo (0): 33.772 (30%)
Total:       113.088
```

**Por que ignorar rating 3?**
- Rating 3 é ambíguo (neutro ou misto)
- Dificulta aprendizado do modelo
- Foco em sentimentos claros (muito bom ou muito ruim)

---

### Modelo 2: Sugestão

#### Classe Positiva (Sugestão = 1)
```python
# Carregar sugestões do arquivo
with open('sugestoes.txt', 'r', encoding='latin1') as f:
    linhas = [linha.strip() for linha in f.readlines()]

df_sugestoes = pd.DataFrame(linhas, columns=['review_text'])
df_sugestoes['is_suggestion'] = 1

# Total: 1.506 sugestões
```

#### Classe Negativa (Sugestão = 0)

**Problema**: B2W contém algumas sugestões misturadas com opiniões puras.

**Solução**: Filtrar palavras-chave de sugestão

```python
suggestion_keywords = [
    'sugiro', 'sugestão', 'sugestões',
    'poderia', 'poderiam', 'deveria', 'deveriam',
    'recomendo que', 'adicionar', 'melhorar',
    'implementar', 'faltou', 'seria bom se'
]

keyword_pattern = '|'.join(suggestion_keywords)

# Filtrar B2W: manter apenas linhas SEM as palavras-chave
df_opinioes_limpas = df_b2w[
    ~df_b2w['review_text'].str.contains(
        keyword_pattern,
        case=False,
        na=False
    )
].copy()

df_opinioes_limpas['is_suggestion'] = 0
```

**Balanceamento**:
```python
# Amostrar mesma quantidade de não-sugestões
df_opinioes_sample = df_opinioes_limpas.sample(
    n=len(df_sugestoes),  # 1.506
    random_state=42
)

# Combinar
df_para_sugestao = pd.concat([
    df_sugestoes,
    df_opinioes_sample
])

# Embaralhar
df_para_sugestao = df_para_sugestao.sample(frac=1, random_state=42)
```

**Distribuição Final**:
```
Sugestão (1):     1.506 (50%)
Não-Sugestão (0): 1.506 (50%)
Total:            3.012 (balanceado)
```

---

## ARQUITETURA DOS MODELOS

### Modelo 1: Classificação de Sentimento

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline_sentimento = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),    # Unigramas + Bigramas
        max_features=50000,     # Top 50k features
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words=None         # Não remove stopwords (contexto)
    )),
    ('clf', LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        C=1.0
    ))
])
```

### Modelo 2: Detecção de Sugestão

```python
pipeline_sugestao = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000
    )),
    ('clf', LogisticRegression(
        random_state=42,
        max_iter=1000
    ))
])
```

**Mesma arquitetura, dados diferentes!**

---

## TREINAMENTO

### Split de Dados

```python
from sklearn.model_selection import train_test_split

# Modelo 1: Sentimento
X_train, X_test, y_train, y_test = train_test_split(
    X_sentimento,
    y_sentimento,
    test_size=0.2,
    random_state=42,
    stratify=y_sentimento  # Mantém proporção de classes
)

# Modelo 2: Sugestão
X_train, X_test, y_train, y_test = train_test_split(
    X_sugestao,
    y_sugestao,
    test_size=0.2,
    random_state=42,
    stratify=y_sugestao
)
```

### Treino

```python
# Treinar
pipeline_sentimento.fit(X_train, y_train)
pipeline_sugestao.fit(X_train, y_train)

# Salvar
import joblib
joblib.dump(pipeline_sentimento, 'models/sentiment_pipeline.pkl')
joblib.dump(pipeline_sugestao, 'models/suggestion_pipeline.pkl')
```

---

## RESULTADOS

### Modelo 1: Sentimento

**Classification Report**:
```
              precision    recall  f1-score   support

    Negativo       0.93      0.91      0.92      6755
    Positivo       0.96      0.97      0.97     15863

    accuracy                           0.95     22618
   macro avg       0.94      0.94      0.94     22618
weighted avg       0.95      0.95      0.95     22618
```

**Matriz de Confusão**:
```
                 Predito
              Neg     Pos
Real  Neg   [6146]  [609]
      Pos   [476] [15387]
```

**Análise**:
- ✅ Alta precision em Positivo (0.96): Poucos falsos positivos
- ✅ Alta recall em Positivo (0.97): Captura maioria dos positivos
- ⚠️ Recall Negativo (0.91): 9% dos negativos classificados como positivos

---

### Modelo 2: Sugestão

**Classification Report**:
```
              precision    recall  f1-score   support

Não-Sugestão       0.97      0.99      0.98       302
    Sugestão       0.99      0.97      0.98       301

    accuracy                           0.98       603
   macro avg       0.98      0.98      0.98       603
weighted avg       0.98      0.98      0.98       603
```

**Matriz de Confusão**:
```
                 Predito
              Não    Sim
Real  Não   [299]   [3]
      Sim   [9]   [292]
```

**Análise**:
- ✅ Accuracy 98%: Excelente
- ✅ Precision/Recall balanceados
- ✅ Pouquíssimos erros (12 em 603)

---

## DEPLOYMENT FLASK

### app.py - Backend

```python
from flask import Flask, render_template, request
from joblib import load
import os

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta'

# Carregar modelos (uma vez na inicialização)
MODELS_DIR = 'models'
pipeline_sentimento = load(os.path.join(MODELS_DIR, 'sentiment_pipeline.pkl'))
pipeline_sugestao = load(os.path.join(MODELS_DIR, 'suggestion_pipeline.pkl'))

@app.route('/')
def home():
    """Página inicial com formulário"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Processa predição dos 2 modelos"""
    # 1. Receber texto
    message = request.form.get('message', '').strip()
    
    if not message:
        return render_template('index.html', 
                             error="Por favor, digite um texto.")
    
    # 2. Modelo 1: Sentimento
    sentiment_code = pipeline_sentimento.predict([message])[0]
    sentiment = 'Positivo' if sentiment_code == 1 else 'Negativo'
    
    # Confiança (probabilidade da classe prevista)
    sentiment_proba = pipeline_sentimento.predict_proba([message])[0]
    sentiment_confidence = round(max(sentiment_proba) * 100, 2)
    
    # 3. Modelo 2: Sugestão
    suggestion_code = pipeline_sugestao.predict([message])[0]
    is_suggestion = 'Sim' if suggestion_code == 1 else 'Não'
    
    suggestion_proba = pipeline_sugestao.predict_proba([message])[0]
    suggestion_confidence = round(max(suggestion_proba) * 100, 2)
    
    # 4. Retornar resultado
    return render_template('resultado.html',
                         review=message,
                         sentiment_prediction=sentiment,
                         sentiment_confidence=sentiment_confidence,
                         is_suggestion=is_suggestion,
                         suggestion_confidence=suggestion_confidence)

if __name__ == '__main__':
    app.run(debug=True)
```

### templates/index.html

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Análise de Feedback</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <h1>📊 Análise de Feedback</h1>
        <p>Insira um feedback para análise automática</p>
        
        <form method="POST" action="/predict">
            <textarea name="message" rows="5" 
                      placeholder="Ex: Adorei o produto! Sugiro adicionar mais cores..."
                      required></textarea>
            <button type="submit">Analisar</button>
        </form>
        
        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
```

### templates/resultado.html

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Resultado da Análise</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Resultado da Análise</h1>
        
        <div class="feedback-text">
            <h3>Feedback analisado:</h3>
            <p>{{ review }}</p>
        </div>
        
        <div class="results">
            <!-- Sentimento -->
            <div class="result-card">
                <h3>💬 Sentimento</h3>
                <p class="prediction {{ 'positive' if sentiment_prediction == 'Positivo' else 'negative' }}">
                    {{ sentiment_prediction }}
                </p>
                <p class="confidence">Confiança: {{ sentiment_confidence }}%</p>
            </div>
            
            <!-- Sugestão -->
            <div class="result-card">
                <h3>💡 Contém Sugestão?</h3>
                <p class="prediction {{ 'suggestion' if is_suggestion == 'Sim' else 'no-suggestion' }}">
                    {{ is_suggestion }}
                </p>
                <p class="confidence">Confiança: {{ suggestion_confidence }}%</p>
            </div>
        </div>
        
        <a href="/" class="back-link">← Analisar outro feedback</a>
    </div>
</body>
</html>
```

---

## ANÁLISE TÉCNICA

### Por Que TF-IDF Funciona Bem?

**Sentimento**:
```python
# Palavras importantes (alto TF-IDF)
Positivo: ["excelente", "adorei", "recomendo", "perfeito"]
Negativo: ["péssimo", "horrível", "decepcionado", "ruim"]
```

**Sugestão**:
```python
# Palavras-chave com alto peso
Sugestão: ["sugiro", "poderia", "deveria", "melhorar", "implementar"]
Opinião:  ["gostei", "adorei", "péssimo", "horrível"]
```

### N-gramas Capturam Contexto

```python
# Exemplo: "não recomendo"
Unigrama: ["não", "recomendo"]          # Ambíguo
Bigrama:  ["não recomendo"]              # Claro (negativo)

# TF-IDF aprende que "não recomendo" tem peso negativo alto
```

### Por Que Logistic Regression?

- ✅ Rápido para treinar e prever
- ✅ Funciona bem com TF-IDF (alta dimensionalidade)
- ✅ Probabilidades calibradas (confiança confiável)
- ✅ Interpretável (coeficientes = importância de palavras)

**Alternativas testadas**:
- SVM: Similar, mas mais lento
- Naive Bayes: ~90% accuracy (pior)
- Random Forest: ~92% accuracy (mais lento)

---

## INTERPRETABILIDADE

### Top Features por Classe

```python
# Obter coeficientes do modelo
tfidf = pipeline_sentimento.named_steps['tfidf']
clf = pipeline_sentimento.named_steps['clf']

feature_names = tfidf.get_feature_names_out()
coef = clf.coef_[0]

# Top 10 palavras Positivas
top_positive = sorted(zip(feature_names, coef), key=lambda x: x[1], reverse=True)[:10]
# [('excelente', 2.45), ('adorei', 2.31), ('perfeito', 2.18), ...]

# Top 10 palavras Negativas
top_negative = sorted(zip(feature_names, coef), key=lambda x: x[1])[:10]
# [('péssimo', -2.67), ('horrível', -2.43), ('ruim', -2.21), ...]
```

---

## LIMITAÇÕES

### 1. Ironia/Sarcasmo

**Exemplo**:
```
"Que produto maravilhoso! [sarcasmo] Chegou quebrado."
→ Modelo prevê: Positivo (errado)
```

**Solução**: Modelos mais avançados (BERT)

### 2. Sugestões Implícitas

**Exemplo**:
```
"Falta uma opção de filtro por preço."
→ Modelo prevê: Não-sugestão (errado)
→ Motivo: Sem palavras-chave ("sugiro", "poderia")
```

**Solução**: Ampliar dataset de sugestões implícitas

### 3. Textos Muito Curtos

**Exemplo**:
```
"Bom"
→ Pouca informação para modelo decidir com confiança
```

### 4. Ambiguidade

**Exemplo**:
```
"Produto OK, mas poderia ser melhor"
→ Sentimento: Neutro (modelo tende para Positivo ou Negativo)
→ Sugestão: Implícita
```

---

## MELHORIAS FUTURAS

### Dados
```python
# Adicionar classe Neutro no sentimento
classes = ['Negativo', 'Neutro', 'Positivo']

# Expandir dataset de sugestões (10k+)
# Incluir sugestões implícitas

# Validação humana (5-10% do dataset)
```

### Modelos
```python
# Testar BERT (transformers)
from transformers import BertTokenizer, BertForSequenceClassification

# Ensemble
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier([
    ('lr', LogisticRegression()),
    ('svm', LinearSVC()),
    ('rf', RandomForestClassifier())
])
```

### Features
```python
# Adicionar features extras
- Comprimento do texto
- Presença de pontos de exclamação
- Ratio de letras maiúsculas (CAPS LOCK = raiva?)
- Número de emojis
```

---

## FAQ TÉCNICO

**Q: Por que não usar um único modelo multi-label?**
A: 2 modelos binários são mais simples, modulares e performam melhor com datasets desbalanceados.

**Q: Por que max_features=50000?**
A: Compromisso entre capturar vocabulário rico e evitar overfit. Testado: 10k (pior), 30k (similar), 100k (overfit).

**Q: Como lidar com palavras fora do vocabulário?**
A: TF-IDF ignora automaticamente. Em produção, considerar subword tokenization (BPE).

**Q: Por que não remover stopwords?**
A: Stopwords podem ter contexto: "não gostei" vs "gostei". TF-IDF já penaliza palavras comuns via IDF.

**Q: Como retreinar com novos dados?**
```python
# Carregar modelo existente
pipeline = load('models/sentiment_pipeline.pkl')

# Treinar com novos dados
pipeline.fit(X_new, y_new)

# Salvar
joblib.dump(pipeline, 'models/sentiment_pipeline_v2.pkl')
```

**Q: Como deployar em produção?**
```python
# Opção 1: Heroku
# Adicionar Procfile: web: gunicorn app:app

# Opção 2: Docker
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

---

## TAGS DE BUSCA

`#nlp-classico` `#analise-sentimento` `#classificacao-texto` `#tfidf` `#logistic-regression` `#flask` `#dupla-classificacao` `#feedback-analysis` `#b2w-reviews` `#sklearn` `#portuguese-nlp`

---

**Versão**: 1.0  
**Compatibilidade**: Python 3.7+, scikit-learn 1.0+, Flask 2.0+  
**Uso recomendado**: Análise de feedbacks, classificação de texto em português, baseline para projetos NLP
