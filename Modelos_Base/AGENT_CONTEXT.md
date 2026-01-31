# AGENT_CONTEXT.md - Modelos Base de NLP Clássico

> **Propósito**: Contexto técnico dos 3 notebooks de modelos prontos  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Templates de modelos para classificação de texto

## RESUMO EXECUTIVO

**Objetivo**: Fornecer modelos prontos e otimizados para classificação de texto  
**Notebooks**: 3 notebooks com código de produção  
**Modelos**: Naive Bayes, SVM, comparações  
**Performance**: 85-92% accuracy típico  
**Uso**: Copiar, adaptar, deployar  
**Diferencial**: Código completo testado, não apenas conceitos

---

## NOTEBOOK 1: naive_bayes_sentimentos.ipynb

### Objetivo
Template completo de Naive Bayes para análise de sentimento.

### Por Que Naive Bayes para NLP?

#### Vantagens Matemáticas
```
P(classe|documento) = P(documento|classe) × P(classe) / P(documento)

Naive Assumption: Features são independentes
P(doc|classe) = P(palavra1|classe) × P(palavra2|classe) × ...

Isso simplifica MUITO o cálculo!
```

#### Vantagens Práticas
- ⚡ **Rápido**: Treino e predição em milissegundos
- 📊 **Poucos dados**: Funciona com 100-1000 exemplos
- 🎯 **Baseline forte**: 85-88% accuracy típico
- 🔍 **Interpretável**: Pode ver probabilidades por palavra

### MultinomialNB - Específico para Texto

```python
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(
    alpha=1.0,        # Laplace smoothing
    fit_prior=True    # Aprende probabilidade das classes
)
```

#### Alpha (Laplace Smoothing)
```python
# Problema: Palavra nunca vista
P("palavranova"|positivo) = 0 / total  # Divisão por zero!

# Solução: Adicionar alpha
P("palavranova"|positivo) = (0 + alpha) / (total + alpha*|V|)

# alpha=1.0: Smoothing padrão
# alpha=0.1: Menos smoothing (mais confiante)
# alpha=10.0: Mais smoothing (mais cauteloso)
```

### Pipeline Completo

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dados
textos = ["Adorei!", "Péssimo!", ...]
labels = ["positivo", "negativo", ...]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    textos, labels, test_size=0.2, random_state=42, stratify=labels
)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,1),  # Apenas unigramas (NB funciona bem assim)
        min_df=2
    )),
    ('nb', MultinomialNB(alpha=1.0))
])

# Treinar
pipeline.fit(X_train, y_train)

# Avaliar
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Accuracy típica: 85-88%
```

### Interpretabilidade - Ver Features Importantes

```python
# Após treinar
tfidf = pipeline.named_steps['tfidf']
nb = pipeline.named_steps['nb']

# Features (palavras)
feature_names = tfidf.get_feature_names_out()

# Log probabilidades por classe
log_probs = nb.feature_log_prob_

# Top 10 palavras por classe
for i, classe in enumerate(nb.classes_):
    top_indices = log_probs[i].argsort()[-10:][::-1]
    top_features = [feature_names[idx] for idx in top_indices]
    print(f"\nTop palavras para '{classe}':")
    print(top_features)
```

### Quando Usar Naive Bayes

**✅ Use Naive Bayes quando**:
- Precisa de baseline rápido
- Tem poucos dados (<10k exemplos)
- Quer interpretabilidade
- Velocidade é crítica
- Classificação de spam, sentimento simples

**❌ Não use Naive Bayes quando**:
- Features são muito correlacionadas
- Precisa capturar interações complexas
- Accuracy <85% não é aceitável
- Tem muitos dados (use SVM ou Deep Learning)

---

## NOTEBOOK 2: classificacao_texto_svm.ipynb

### Objetivo
Template de SVM para classificação multi-classe com alta performance.

### Por Que SVM para NLP?

#### Support Vector Machines - Conceito
```
Encontrar hiperplano que melhor separa as classes
maximizando a margem entre elas

     Classe A        |        Classe B
        •            |            ◦
      •   •       MARGEM        ◦  ◦
        •            |            ◦
```

#### Por Que Funciona Bem em Texto?
- ✅ **Alta dimensionalidade**: 10k-100k features não é problema
- ✅ **Espaços esparsos**: TF-IDF é esparso, SVM lida bem
- ✅ **Margens claras**: Textos de classes diferentes geralmente bem separados
- ✅ **Kernel trick**: Pode aprender relações não-lineares

### LinearSVC - Otimizado para Texto

```python
from sklearn.svm import LinearSVC

clf = LinearSVC(
    C=1.0,              # Regularização (inversa)
    max_iter=1000,      # Iterações máximas
    dual=False,         # Primal (se n_samples > n_features)
    random_state=42
)
```

#### Parâmetro C (Regularização)

```python
# C pequeno (0.1): Mais regularização
# → Margin maior, pode underfit
# → Generaliza melhor, menos overfit

# C médio (1.0): Padrão balanceado ✓

# C grande (10.0): Menos regularização
# → Margin menor, pode overfit
# → Accuracy maior no treino
```

### Pipeline Completo

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,     # Mais features que NB
        ngram_range=(1, 2),     # Uni + Bigramas
        min_df=2,
        max_df=0.8,
        sublinear_tf=True       # log(TF)
    )),
    ('svm', LinearSVC(
        C=1.0,
        max_iter=1000,
        random_state=42
    ))
])

# Grid Search (opcional)
param_grid = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'svm__C': [0.1, 1.0, 10.0]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Treinar
grid.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Melhor CV score: {grid.best_score_:.3f}")

# Accuracy típica: 89-92%
```

### Troubleshooting - Convergência

```python
# Problema: ConvergenceWarning
# "Objective did not converge"

# Solução 1: Aumentar iterações
LinearSVC(max_iter=2000)

# Solução 2: Normalizar (já faz TF-IDF)
# TfidfVectorizer aplica L2 norm automaticamente

# Solução 3: Reduzir C
LinearSVC(C=0.1)
```

### SVM vs Naive Bayes

```python
# Experimento típico:
# Dataset: 10k reviews, 2 classes

Naive Bayes:
- Treino: ~1 segundo
- Predição: ~0.1 segundo
- Accuracy: 86%

LinearSVC:
- Treino: ~5 segundos
- Predição: ~0.1 segundo
- Accuracy: 91%

Conclusão: SVM vale o custo extra de treino para +5% accuracy
```

### Quando Usar SVM

**✅ Use SVM quando**:
- Quer melhor accuracy (~90%+)
- Tem >1k exemplos
- Alta dimensionalidade (TF-IDF)
- Classificação multi-classe
- Produção (predição é rápida)

**❌ Não use SVM quando**:
- Tem >1M exemplos (muito lento)
- Precisa de probabilidades calibradas (use LogisticRegression)
- Quer interpretabilidade máxima (use NB)

---

## NOTEBOOK 3: comparativo_tfidf_vs_embeddings.ipynb

### Objetivo
Comparação empírica de 3 abordagens para classificação de texto.

### Experimento 1: TF-IDF + Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Pipeline
pipeline_tfidf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Treinar e avaliar
pipeline_tfidf.fit(X_train, y_train)
acc_tfidf = pipeline_tfidf.score(X_test, y_test)

print(f"TF-IDF Accuracy: {acc_tfidf:.3f}")
# Típico: ~87%
```

**Vantagens**:
- Simples e rápido
- Não precisa treinar embeddings
- Funciona bem para classificação

**Desvantagens**:
- Não captura semântica
- Alta dimensionalidade

---

### Experimento 2: Word2Vec Próprio + Logistic Regression

```python
from gensim.models import Word2Vec
import numpy as np

# 1. Treinar Word2Vec
tokenized = [text.split() for text in X_train]
w2v = Word2Vec(
    sentences=tokenized,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# 2. Vetorizar documentos (média dos vetores)
def doc_vector(text, model):
    words = text.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

X_train_w2v = np.array([doc_vector(t, w2v) for t in X_train])
X_test_w2v = np.array([doc_vector(t, w2v) for t in X_test])

# 3. Treinar
clf_w2v = LogisticRegression(max_iter=1000)
clf_w2v.fit(X_train_w2v, y_train)

acc_w2v = clf_w2v.score(X_test_w2v, y_test)
print(f"Word2Vec Accuracy: {acc_w2v:.3f}")
# Típico: ~85% (menor que TF-IDF se corpus pequeno!)
```

**Vantagens**:
- Captura semântica
- Dimensionalidade baixa (100D)

**Desvantagens**:
- Precisa corpus grande (>1M palavras)
- Perda de informação (média dos vetores)
- Pode ser pior que TF-IDF em datasets pequenos

---

### Experimento 3: Embeddings Pré-treinados + Logistic Regression

```python
from gensim.models import KeyedVectors

# 1. Carregar embeddings pré-treinados
pretrained = KeyedVectors.load_word2vec_format(
    'nilc_skip_s300.txt',  # 300 dimensões
    binary=False
)

# 2. Vetorizar
def doc_vector_pretrained(text, model):
    words = text.split()
    vectors = [model[w] for w in words if w in model]
    if not vectors:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

X_train_pre = np.array([doc_vector_pretrained(t, pretrained) for t in X_train])
X_test_pre = np.array([doc_vector_pretrained(t, pretrained) for t in X_test])

# 3. Treinar
clf_pre = LogisticRegression(max_iter=1000)
clf_pre.fit(X_train_pre, y_train)

acc_pre = clf_pre.score(X_test_pre, y_test)
print(f"Pretrained Accuracy: {acc_pre:.3f}")
# Típico: ~90% (melhor!)
```

**Vantagens**:
- Não precisa treinar embeddings
- Treinado em bilhões de palavras
- Captura semântica
- **Geralmente o melhor para datasets pequenos-médios**

**Desvantagens**:
- Arquivo grande (~1-5 GB)
- Domínio geral (não específico)

---

### Comparação Final

```python
import matplotlib.pyplot as plt

resultados = {
    'TF-IDF': acc_tfidf,
    'Word2Vec\n(próprio)': acc_w2v,
    'Pré-treinado': acc_pre
}

plt.figure(figsize=(10, 6))
bars = plt.bar(resultados.keys(), resultados.values(), 
               color=['#3498db', '#e74c3c', '#2ecc71'])
plt.ylabel('Accuracy')
plt.title('Comparação de Representações de Texto')
plt.ylim(0.5, 1.0)
plt.axhline(y=0.9, color='gray', linestyle='--', label='90% threshold')
plt.legend()

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center', va='bottom')

plt.show()
```

**Resultado Típico**:
```
TF-IDF:          87% ⭐⭐⭐
Word2Vec próprio: 85% ⭐⭐
Pré-treinado:     90% ⭐⭐⭐⭐ ← Melhor!
```

---

### Quando Usar Cada Abordagem

| Abordagem | Dataset | Accuracy | Velocidade | Memória |
|-----------|---------|----------|------------|---------|
| **TF-IDF** | Qualquer | 87% | ⚡⚡⚡ | Baixa |
| **Word2Vec Próprio** | >100k docs | 85% | ⚡ | Baixa |
| **Pré-treinado** | <10k docs | 90% | ⚡⚡ | Alta (2GB+) |

**Regra prática**:
```python
if dataset_size < 10000:
    use_pretrained_embeddings()  # Melhor accuracy
elif dataset_size < 100000:
    use_tfidf()  # Simples e eficaz
else:
    train_word2vec()  # Aprende domínio específico
```

---

## CÓDIGO DE REFERÊNCIA COMPLETO

### Template Produção - Classificação de Texto

```python
# ===== IMPORTS =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===== CARREGAR DADOS =====
df = pd.read_csv('dados.csv')
df.dropna(subset=['texto', 'categoria'], inplace=True)

X = df['texto']
y = df['categoria']

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== PIPELINE =====
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True
    )),
    ('clf', LinearSVC(
        C=1.0,
        max_iter=1000,
        random_state=42
    ))
])

# ===== GRID SEARCH (opcional) =====
param_grid = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1.0, 10.0]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

# ===== TREINAR =====
print("Treinando...")
grid.fit(X_train, y_train)

print(f"\nMelhores parâmetros: {grid.best_params_}")
print(f"Melhor CV score: {grid.best_score_:.3f}")

# ===== AVALIAR =====
y_pred = grid.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===== MATRIZ DE CONFUSÃO =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=grid.classes_,
            yticklabels=grid.classes_)
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ===== SALVAR MODELO =====
joblib.dump(grid.best_estimator_, 'modelo_texto.pkl')
print("\nModelo salvo em 'modelo_texto.pkl'")

# ===== USAR MODELO =====
model = joblib.load('modelo_texto.pkl')
novo_texto = "Texto de exemplo para classificar"
predicao = model.predict([novo_texto])
print(f"\nPredição: {predicao[0]}")
```

**Este código é pronto para produção!**

---

## HIPERPARÂMETROS RECOMENDADOS

### TF-IDF
```python
TfidfVectorizer(
    max_features=10000,      # 10k para maioria dos casos
    ngram_range=(1, 2),      # Uni + Bigramas
    min_df=2,                # Ignora palavras em <2 docs
    max_df=0.8,              # Ignora palavras em >80% docs
    sublinear_tf=True,       # log(TF) em vez de TF
    strip_accents='unicode', # Remove acentos
    lowercase=True,          # Minúsculas
    stop_words=None          # Não remove (TF-IDF já filtra)
)
```

### Multinomial NB
```python
MultinomialNB(
    alpha=1.0,          # Laplace smoothing padrão
    fit_prior=True      # Aprende prior das classes
)
```

### LinearSVC
```python
LinearSVC(
    C=1.0,              # Regularização padrão
    max_iter=1000,      # Suficiente para maioria
    dual=False,         # Primal se n_samples > n_features
    class_weight=None,  # Ou 'balanced' se desbalanceado
    random_state=42
)
```

### Logistic Regression
```python
LogisticRegression(
    C=1.0,                  # Regularização inversa
    max_iter=1000,
    solver='lbfgs',         # Padrão e eficiente
    multi_class='ovr',      # One-vs-Rest
    class_weight=None,      # Ou 'balanced'
    random_state=42
)
```

---

## CHECKLIST DE CONCLUSÃO

- [ ] Treinei Naive Bayes
- [ ] Treinei LinearSVC
- [ ] Comparei TF-IDF vs Embeddings
- [ ] Sei escolher modelo por tarefa
- [ ] Entendo hiperparâmetros principais
- [ ] Criei pipeline completo pronto para produção

---

## TAGS DE BUSCA

`#modelos-classicos` `#naive-bayes` `#svm` `#logistic-regression` `#tfidf` `#word2vec` `#embeddings` `#classificacao-texto` `#sklearn` `#pipeline` `#grid-search`

---

**Versão**: 1.0  
**Compatibilidade**: scikit-learn 1.0+  
**Uso recomendado**: Templates de produção, baseline rápido, comparação de modelos
