# Modelos Base - NLP Clássico

## 📌 Sobre

Esta pasta contém **modelos clássicos de Machine Learning** aplicados a problemas de NLP, com código pronto e testado para classificação de texto.

**Objetivo**: Fornecer templates completos de modelos que funcionam bem em tarefas de NLP clássico.

---

## 🎯 Diferença: Fundamentos vs Modelos_Base

| Aspecto | Fundamentos | Modelos_Base |
|---------|-------------|--------------|
| **Foco** | Técnicas de NLP (TF-IDF, embeddings) | Modelos de ML prontos |
| **Conteúdo** | Como vetorizar texto | Como classificar texto |
| **Uso** | Aprender conceitos | Copiar e adaptar código |
| **Exemplos** | Pequenos e didáticos | Completos e otimizados |

---

## 📂 Arquivos Disponíveis

### 📘 **naive_bayes_sentimentos.ipynb** ⭐

**Descrição**: Classificação de sentimentos usando Naive Bayes

**Problema**: Classificar reviews como positivos ou negativos

**Por Que Naive Bayes para NLP?**
- ✅ Rápido para treinar e prever
- ✅ Funciona bem com texto (alta dimensionalidade)
- ✅ Baseline forte para classificação de texto
- ✅ Requer poucos dados

#### Pipeline Completo

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dados
textos = ["Adorei o filme!", "Péssimo produto", ...]
labels = ["positivo", "negativo", ...]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    textos, labels, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('nb', MultinomialNB(alpha=1.0))
])

# Treinar
pipeline.fit(X_train, y_train)

# Avaliar
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

#### Resultado Esperado
```
              precision    recall  f1-score
positivo          0.87      0.85      0.86
negativo          0.84      0.86      0.85

accuracy                              0.85
```

#### Hiperparâmetros

**alpha (Laplace Smoothing)**:
```python
# alpha=1.0: Smoothing padrão (recomendado)
# alpha=0.1: Menos smoothing (mais confiante)
# alpha=10.0: Mais smoothing (mais cauteloso)

nb = MultinomialNB(alpha=1.0)
```

#### Quando Usar Naive Bayes?
- ✅ Baseline rápido
- ✅ Poucos dados de treino
- ✅ Classificação de texto (spam, sentimento)
- ❌ Não quando features são correlacionadas (viola "naive")

---

### 📘 **classificacao_texto_svm.ipynb** ⭐

**Descrição**: Classificação de texto com Support Vector Machines

**Problema**: Classificar notícias em categorias (política, esporte, tecnologia)

**Por Que SVM para NLP?**
- ✅ Excelente com alta dimensionalidade
- ✅ Funciona bem em espaços esparsos (TF-IDF)
- ✅ Margins claros entre classes
- ✅ Melhor que Naive Bayes em muitos casos

#### Pipeline Completo

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Uni + Bigramas
        min_df=2
    )),
    ('svm', LinearSVC(
        C=1.0,
        max_iter=1000,
        random_state=42
    ))
])

# Treinar
pipeline.fit(X_train, y_train)

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Avaliar
accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.3f}")
```

#### Resultado Esperado
```
CV Score: 0.892 (+/- 0.012)
Test Accuracy: 0.897
```

#### Hiperparâmetros

**C (Regularização)**:
```python
# C pequeno (0.1): Mais regularização, margin maior
# C médio (1.0): Padrão, balanceado
# C grande (10.0): Menos regularização, margin menor

svm = LinearSVC(C=1.0)
```

**max_iter**:
```python
# Aumentar se não convergir
svm = LinearSVC(max_iter=2000)
```

#### SVM vs Naive Bayes

| Aspecto | Naive Bayes | SVM |
|---------|-------------|-----|
| **Velocidade** | Mais rápido | Mais lento |
| **Accuracy** | Boa | Melhor |
| **Poucos dados** | Funciona | Pode overfit |
| **Alta dim** | Funciona bem | Funciona muito bem |

---

### 📘 **comparativo_tfidf_vs_embeddings.ipynb** ⭐⭐

**Descrição**: Comparação direta entre TF-IDF e Word Embeddings

**Objetivo**: Descobrir qual representação funciona melhor para seu problema

#### Experimento 1: TF-IDF + Logistic Regression

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# TF-IDF
vectorizer_tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# Modelo
clf_tfidf = LogisticRegression(max_iter=1000)
clf_tfidf.fit(X_train_tfidf, y_train)

# Avaliar
acc_tfidf = clf_tfidf.score(X_test_tfidf, y_test)
print(f"TF-IDF Accuracy: {acc_tfidf:.3f}")
```

#### Experimento 2: Word2Vec + Logistic Regression

```python
from gensim.models import Word2Vec
import numpy as np

# Treinar Word2Vec
tokenized_texts = [text.split() for text in X_train]
w2v_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# Vetorizar documentos (média dos vetores das palavras)
def document_vector(text, model):
    words = text.split()
    word_vectors = [model.wv[w] for w in words if w in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

X_train_w2v = np.array([document_vector(t, w2v_model) for t in X_train])
X_test_w2v = np.array([document_vector(t, w2v_model) for t in X_test])

# Modelo
clf_w2v = LogisticRegression(max_iter=1000)
clf_w2v.fit(X_train_w2v, y_train)

# Avaliar
acc_w2v = clf_w2v.score(X_test_w2v, y_test)
print(f"Word2Vec Accuracy: {acc_w2v:.3f}")
```

#### Experimento 3: Embeddings Pré-treinados

```python
from gensim.models import KeyedVectors

# Carregar embeddings pré-treinados (ex: NILC)
pretrained = KeyedVectors.load_word2vec_format(
    'embeddings/skip_s300.txt',
    binary=False
)

def document_vector_pretrained(text, model):
    words = text.split()
    word_vectors = [model[w] for w in words if w in model]
    if not word_vectors:
        return np.zeros(300)  # Dimensão do modelo
    return np.mean(word_vectors, axis=0)

X_train_pre = np.array([document_vector_pretrained(t, pretrained) for t in X_train])
X_test_pre = np.array([document_vector_pretrained(t, pretrained) for t in X_test])

# Modelo
clf_pre = LogisticRegression(max_iter=1000)
clf_pre.fit(X_train_pre, y_train)

# Avaliar
acc_pre = clf_pre.score(X_test_pre, y_test)
print(f"Pretrained Accuracy: {acc_pre:.3f}")
```

#### Comparação de Resultados

```python
import matplotlib.pyplot as plt

resultados = {
    'TF-IDF': acc_tfidf,
    'Word2Vec': acc_w2v,
    'Pretrained': acc_pre
}

plt.figure(figsize=(10, 6))
plt.bar(resultados.keys(), resultados.values())
plt.ylabel('Accuracy')
plt.title('Comparação de Representações')
plt.ylim(0.5, 1.0)
plt.axhline(y=0.9, color='r', linestyle='--', label='90% baseline')
plt.legend()
plt.show()
```

#### Resultado Típico

```
TF-IDF Accuracy:       0.892
Word2Vec Accuracy:     0.857
Pretrained Accuracy:   0.903  ← Melhor!
```

**Por quê?**
- TF-IDF captura keywords específicas
- Word2Vec treinado pode ter pouco corpus
- Pré-treinado tem qualidade superior (bilhões de palavras)

#### Quando Usar Cada Um?

| Representação | Quando Usar |
|---------------|-------------|
| **TF-IDF** | Keywords importantes, documentos formais, baseline |
| **Word2Vec Próprio** | Corpus grande (>1M palavras), domínio específico |
| **Pré-treinado** | Poucos dados, domínio geral, melhor qualidade |

---

## 📊 Comparação de Modelos

### Performance Típica em Classificação de Texto

| Modelo | Accuracy | Velocidade Treino | Velocidade Predição |
|--------|----------|-------------------|---------------------|
| **Naive Bayes** | 85-88% | ⚡ Muito rápido | ⚡ Muito rápido |
| **Logistic Reg** | 87-90% | ⚡ Rápido | ⚡ Rápido |
| **LinearSVC** | 89-92% | 🐢 Médio | ⚡ Rápido |
| **Random Forest** | 85-88% | 🐢 Lento | 🐢 Médio |

### Quando Usar Cada Modelo?

**Naive Bayes**:
- ✅ Baseline rápido
- ✅ Poucos dados
- ✅ Spam detection
- ❌ Features correlacionadas

**LinearSVC**:
- ✅ Melhor accuracy
- ✅ Alta dimensionalidade
- ✅ Classificação multi-classe
- ❌ Mais lento que NB

**Logistic Regression**:
- ✅ Interpretável
- ✅ Probabilidades calibradas
- ✅ Balanced entre velocidade e accuracy
- ❌ Linear (não captura interações)

---

## 💻 Código Completo de Referência

### Template Completo: Classificação de Texto

```python
# ===== IMPORTS =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===== PRÉ-PROCESSAMENTO =====
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_pt = set(stopwords.words('portuguese'))

def preprocessar(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    palavras = [p for p in texto.split() if p not in stopwords_pt]
    return ' '.join(palavras)

# ===== CARREGAR DADOS =====
df = pd.read_csv('dados.csv')
df['texto_limpo'] = df['texto'].apply(preprocessar)

# ===== SPLIT =====
X = df['texto_limpo']
y = df['categoria']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== PIPELINE =====
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )),
    ('clf', LinearSVC(
        C=1.0,
        max_iter=1000,
        random_state=42
    ))
])

# ===== GRID SEARCH (Opcional) =====
param_grid = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1.0, 10.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Melhor score CV: {grid.best_score_:.3f}")

# ===== AVALIAR =====
y_pred = grid.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===== MATRIZ DE CONFUSÃO =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=grid.classes_,
            yticklabels=grid.classes_)
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()

# ===== SALVAR MODELO =====
import joblib
joblib.dump(grid.best_estimator_, 'modelo_texto.pkl')
print("Modelo salvo!")
```

---

## 🎯 Hiperparâmetros Recomendados

### TF-IDF

```python
TfidfVectorizer(
    max_features=10000,     # Top 10k palavras
    ngram_range=(1, 2),     # Uni + Bigramas
    min_df=2,               # Palavra em ≥2 docs
    max_df=0.8,             # Ignora >80% docs
    sublinear_tf=True       # log(TF) em vez de TF
)
```

### Naive Bayes

```python
MultinomialNB(
    alpha=1.0,              # Laplace smoothing padrão
    fit_prior=True          # Aprende prior das classes
)
```

### LinearSVC

```python
LinearSVC(
    C=1.0,                  # Regularização padrão
    max_iter=1000,          # Iterações máximas
    dual=False,             # Primal (se n_samples > n_features)
    random_state=42
)
```

### Logistic Regression

```python
LogisticRegression(
    C=1.0,                  # Regularização inversa
    max_iter=1000,
    solver='lbfgs',         # Solver padrão
    multi_class='ovr',      # One-vs-Rest
    random_state=42
)
```

---

## 🔧 Troubleshooting

### Problema: LinearSVC não converge

**Solução**:
```python
# Aumentar iterações
LinearSVC(max_iter=2000)

# Ou normalizar features
from sklearn.preprocessing import StandardScaler
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('scaler', StandardScaler(with_mean=False)),  # Sparse-safe
    ('svm', LinearSVC())
])
```

### Problema: Word2Vec KeyError (palavra não existe)

**Solução**:
```python
def document_vector_safe(text, model):
    words = text.split()
    vectors = []
    for w in words:
        if w in model.wv:  # Verifica se existe
            vectors.append(model.wv[w])
    
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)
```

### Problema: Accuracy muito baixa (<60%)

**Causas e Soluções**:
1. **Pré-processamento ruim**
   - Verificar se removeu stopwords
   - Testar com/sem stemming

2. **Dados desbalanceados**
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   
   class_weights = compute_class_weight(
       'balanced',
       classes=np.unique(y_train),
       y=y_train
   )
   clf = LinearSVC(class_weight='balanced')
   ```

3. **Features insuficientes**
   ```python
   # Aumentar max_features
   TfidfVectorizer(max_features=20000)
   ```

---

## 📈 Experimentos Sugeridos

### Experimento 1: N-gramas
```python
# Testar diferentes n-gramas
for ngram in [(1,1), (1,2), (1,3), (2,2)]:
    vectorizer = TfidfVectorizer(ngram_range=ngram)
    # Treinar e avaliar
```

### Experimento 2: Modelos
```python
modelos = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Reg': LogisticRegression(),
    'LinearSVC': LinearSVC()
}

for nome, clf in modelos.items():
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', clf)])
    score = cross_val_score(pipeline, X, y, cv=5).mean()
    print(f"{nome}: {score:.3f}")
```

### Experimento 3: Tamanho do Corpus
```python
# Treinar com diferentes tamanhos
for size in [100, 500, 1000, 5000]:
    X_sample = X_train[:size]
    y_sample = y_train[:size]
    # Treinar e avaliar
```

---

## ✅ Checklist de Projeto

- [ ] Pré-processamento aplicado
- [ ] TF-IDF configurado (max_features, ngram_range)
- [ ] Modelo escolhido e treinado
- [ ] Cross-validation executada
- [ ] Teste em conjunto separado
- [ ] Classification report analisado
- [ ] Matriz de confusão plotada
- [ ] Modelo salvo (joblib)

---

## 🚀 Próximos Passos

Após dominar Modelos_Base:

1. **Ir para Projetos/** - Aplicações deployadas
2. **Experimentar ensemble** - Combinar modelos
3. **Explorar Deep Learning** - RNNs, Transformers (EAI_05)

---

**Lembre-se**: Modelos clássicos (SVM + TF-IDF) ainda são baseline forte. Sempre teste antes de usar modelos complexos!

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_04*
