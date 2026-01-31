# AGENT_CONTEXT.md - EAI_04 NLP Clássico (Módulo Completo)

> **Propósito**: Visão técnica completa do módulo EAI_04  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Módulo educacional com 3 seções progressivas

## RESUMO EXECUTIVO

**Objetivo**: Ensinar NLP Clássico de forma prática e aplicável  
**Estrutura**: Fundamentos → Modelos → Projetos  
**Notebooks**: 9 notebooks conceituais  
**Projetos**: 2 aplicações Flask deployadas  
**Técnicas**: BoW, TF-IDF, Word2Vec, Sentence Transformers  
**Modelos**: Naive Bayes, SVM, Logistic Regression  
**Diferencial**: Progressão incremental com código de produção

---

## DESIGN PEDAGÓGICO

### Abordagem: Learning by Doing

```
Teoria → Prática Guiada → Aplicação Autônoma

Fundamentos (70% teoria, 30% prática)
    ↓ Construir base sólida
Modelos Base (30% teoria, 70% prática)
    ↓ Templates prontos
Projetos (10% teoria, 90% prática)
    ↓ Aplicações reais
```

### Progressão de Complexidade

| Seção | Conceitos | Código | Autonomia | Objetivo |
|-------|-----------|--------|-----------|----------|
| **Fundamentos** | ⭐⭐⭐ | ⭐ | Guiado | Aprender |
| **Modelos Base** | ⭐ | ⭐⭐⭐ | Semi-guiado | Aplicar |
| **Projetos** | ⭐ | ⭐⭐⭐ | Autônomo | Deployar |

---

## SEÇÃO 1: FUNDAMENTOS

### Estrutura Pedagógica

**Objetivo**: Base sólida antes de modelos complexos  
**Método**: Conceito → Código → Exemplo → Exercício

### Notebooks e Objetivos

#### 1. pre_processamento_texto.ipynb
**Objetivo de Aprendizado**: Pipeline de limpeza  
**Conceitos**:
```python
lowercase → remove_punct → tokenize → 
remove_stopwords → stemming → clean_text
```

**Código Chave**:
```python
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    palavras = texto.split()
    palavras = [p for p in palavras if p not in stopwords_pt]
    palavras = [stemmer.stem(p) for p in palavras]
    return ' '.join(palavras)
```

**Conceito-Chave**: "Garbage in, garbage out"

---

#### 2. bow_tfidf.ipynb
**Objetivo de Aprendizado**: Primeira representação vetorial  

**Fórmulas Matemáticas**:
```
BoW: word_count(w, d)

TF(w,d) = count(w,d) / total_words(d)
IDF(w) = log(N / df(w))
TF-IDF(w,d) = TF(w,d) × IDF(w)
```

**Código Chave**:
```python
# BoW
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(textos)

# TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(textos)
```

---

#### 3. representacao_bow_tfidf.ipynb
**Objetivo de Aprendizado**: Otimização de parâmetros  

**Parâmetros Críticos**:
```python
TfidfVectorizer(
    ngram_range=(1,2),  # Captura "não gostei"
    max_df=0.8,         # Remove palavras muito comuns
    min_df=2,           # Remove typos
    max_features=10000  # Controla dimensionalidade
)
```

---

#### 4. word_embeddings.ipynb
**Objetivo de Aprendizado**: Representação semântica  

**Arquiteturas**:
```
CBOW: Contexto → Palavra
Skip-gram: Palavra → Contexto
```

**Código Chave**:
```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=corpus_tokenizado,
    vector_size=100,
    window=5,
    sg=0  # 0=CBOW, 1=Skip-gram
)

# Semântica
model.wv.most_similar('rei')
# [('rainha', 0.89), ...]
```

---

#### 5. pretrained_embeddings.ipynb
**Objetivo de Aprendizado**: Usar embeddings prontos  

**Vantagens**:
- ✅ Não precisa treinar
- ✅ Bilhões de palavras
- ✅ Alta qualidade

**Código Chave**:
```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    'skip_s300.txt',
    binary=False
)
```

---

#### 6. analise_sentimentos.ipynb
**Objetivo de Aprendizado**: Pipeline end-to-end  

**Pipeline Completo**:
```python
Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
```

**Performance Esperada**: ~85-90% accuracy

---

## SEÇÃO 2: MODELOS BASE

### Estrutura Pedagógica

**Objetivo**: Templates de código prontos para produção  
**Método**: Modelo → Hiperparâmetros → Uso → Quando usar

### Notebooks e Arquiteturas

#### 1. naive_bayes_sentimentos.ipynb

**Modelo**: MultinomialNB  

**Por que funciona em NLP**:
```
P(classe|doc) ∝ P(classe) × ∏ P(palavra|classe)

Assumption "naive": palavras independentes
Simplifica cálculo MUITO!
```

**Hiperparâmetros**:
```python
MultinomialNB(
    alpha=1.0  # Laplace smoothing
)

# alpha=0.1: Menos smoothing
# alpha=1.0: Padrão ✓
# alpha=10.0: Mais smoothing
```

**Performance**: ~85-88%  
**Quando usar**: Baseline, poucos dados

---

#### 2. classificacao_texto_svm.ipynb

**Modelo**: LinearSVC  

**Por que funciona em NLP**:
```
Encontra hiperplano com margem máxima
Funciona bem em alta dimensionalidade (10k-100k features)
```

**Hiperparâmetros**:
```python
LinearSVC(
    C=1.0  # Regularização inversa
)

# C=0.1: Mais regularização
# C=1.0: Padrão ✓
# C=10.0: Menos regularização
```

**Performance**: ~89-92%  
**Quando usar**: Melhor accuracy, produção

---

#### 3. comparativo_tfidf_vs_embeddings.ipynb

**Experimento**: 3 abordagens  

**Resultados Típicos**:
```python
TF-IDF + LogReg:              87%
Word2Vec próprio + LogReg:    85%
Embeddings pré-treinados:     90% ← Melhor!
```

**Lição**: Pré-treinados > TF-IDF para datasets pequenos

---

## SEÇÃO 3: PROJETOS

### Estrutura Pedagógica

**Objetivo**: Código de produção deployado  
**Método**: Arquitetura → Implementação → Deploy → Uso

### Projeto 1: Análise de Feedback

#### Arquitetura

```
Sistema de Dupla Classificação:

Input: "Adorei! Sugiro adicionar mais cores"
    ↓
┌───────────┴───────────┐
│                       │
Modelo 1           Modelo 2
Sentimento         Sugestão
    ↓                  ↓
Positivo (95%)     Sim (97%)
    ↓                  ↓
└───────────┬───────────┘
            ↓
    Output Combinado
```

#### Tecnologia

**Modelo 1 - Sentimento**:
```python
Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1,2),
        max_features=50000
    )),
    ('clf', LogisticRegression())
])

Dataset: B2W-Reviews (113k)
Performance: 95% accuracy
```

**Modelo 2 - Sugestão**:
```python
# Mesma arquitetura
Dataset: Sugestões IA + B2W filtrado (3k balanceado)
Performance: 98% accuracy
```

#### Conceito-Chave

**Por que 2 modelos?**
```python
# Abordagem 1 (não usada): Multi-label
classes = [
    'Pos+Sug', 'Pos+NoSug',
    'Neg+Sug', 'Neg+NoSug'
]
# Complexo, requer mais dados

# Abordagem 2 (usada): 2 binários ✓
# Simples, modular, melhor performance
```

---

### Projeto 2: Sistema de Busca FAQs

#### Arquitetura

```
Busca Semântica:

Pergunta Usuário: "Como fazer PIX?"
    ↓
Sentence Transformer (embedding 512D)
    ↓
Similaridade Cosseno com Base (1172 FAQs)
    ↓
Top 3 Resultados (≥50% threshold)
    ↓
1. "Como acesso o PIX?" (87%)
2. "Como cadastrar chave?" (74%)
3. "Qual limite PIX?" (62%)
```

#### Tecnologia

**Modelo**:
```python
SentenceTransformer('distiluse-base-multilingual-cased-v1')
# 512 dimensões
# Multilíngue
# Distilado (rápido)
```

**Busca**:
```python
from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(
    embedding_query,
    embeddings_base
)

# Filtrar por threshold
resultados = [r for r in top_k if sim >= 0.5]
```

#### Conceito-Chave

**Busca Semântica vs Keywords**:
```
Keywords: "fazer" ≠ "realizar" → Miss
Semântica: Entende sinônimos → Hit (87%)
```

---

## COMPARAÇÃO DE TÉCNICAS

### Representações

| Técnica | Dim | Tipo | Semântica | Treino | Uso Típico |
|---------|-----|------|-----------|--------|------------|
| **BoW** | 10k-100k | Esparso | ❌ | Não | Baseline |
| **TF-IDF** | 10k-100k | Esparso | ❌ | Não | Classificação |
| **Word2Vec** | 100-300 | Denso | ✅ | Sim | Similaridade |
| **FastText** | 100-300 | Denso | ✅ | Sim | OOV words |
| **Sentence Transformers** | 512-768 | Denso | ✅✅ | Não | Busca semântica |

### Modelos

| Modelo | Accuracy | Treino | Predição | Interpretável | Produção |
|--------|----------|--------|----------|---------------|----------|
| **Naive Bayes** | 85% | ⚡⚡⚡ | ⚡⚡⚡ | ⭐⭐⭐ | ✅ |
| **Logistic Reg** | 87% | ⚡⚡ | ⚡⚡ | ⭐⭐⭐ | ✅ |
| **LinearSVC** | 90% | ⚡ | ⚡⚡ | ⭐⭐ | ✅ |
| **Random Forest** | 88% | 🐢 | 🐢 | ⭐ | ❌ |

---

## PIPELINE TÍPICO

### Template de Produção

```python
# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

# 2. Dados
df = pd.read_csv('dados.csv')
X = df['texto']
y = df['categoria']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# 4. Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2)
    )),
    ('clf', LinearSVC(C=1.0))
])

# 5. Treinar
pipeline.fit(X_train, y_train)

# 6. Avaliar
print(classification_report(y_test, pipeline.predict(X_test)))

# 7. Salvar
joblib.dump(pipeline, 'modelo.pkl')
```

**Este pipeline resolve 80% dos problemas!**

---

## DEPLOYMENT

### Flask Básico

```python
from flask import Flask, request
import joblib

app = Flask(__name__)
model = joblib.load('modelo.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    texto = request.json['texto']
    pred = model.predict([texto])[0]
    return {'predicao': pred}

if __name__ == '__main__':
    app.run()
```

### Docker

```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

### Heroku

```bash
# Procfile
web: gunicorn app:app

# Deploy
heroku create
git push heroku main
```

---

## TROUBLESHOOTING COMUM

### Problema 1: Accuracy Baixa (<70%)

```python
# Causas:
1. Pré-processamento ruim → Verificar stopwords
2. Dados desbalanceados → class_weight='balanced'
3. Features insuficientes → Aumentar max_features
4. Modelo inadequado → Testar SVM em vez de NB
```

### Problema 2: Modelo Não Converge

```python
# Solução:
LinearSVC(max_iter=2000)  # Aumentar iterações
```

### Problema 3: OOM (Out of Memory)

```python
# Solução:
TfidfVectorizer(max_features=5000)  # Reduzir features
```

---

## PROGRESSÃO PARA NLP MODERNO

### Fundação (EAI_04 - Este Módulo)
```
✅ TF-IDF
✅ Word2Vec
✅ Naive Bayes, SVM
✅ Sentence Transformers (básico)
```

### Próximo Nível (EAI_05)
```
→ Transformers (atenção)
→ BERT, RoBERTa
→ Fine-tuning
→ GPT (generativo)
→ Hugging Face
```

**Base Sólida**: Este módulo é essencial para entender Transformers!

---

## MÉTRICAS DE SUCESSO

### Conhecimento
- [ ] Entende TF-IDF matematicamente
- [ ] Sabe quando usar BoW vs Embeddings
- [ ] Domina pré-processamento
- [ ] Conhece limitações de cada técnica

### Habilidades
- [ ] Treina modelo do zero
- [ ] Cria pipeline completo
- [ ] Compara modelos
- [ ] Usa embeddings pré-treinados

### Aplicação
- [ ] Executou 2 projetos
- [ ] Modificou projeto existente
- [ ] Criou projeto próprio
- [ ] Fez deploy em produção

---

## BIBLIOTECAS DO MÓDULO

```python
# NLP Clássico
nltk              # Stopwords, stemming
scikit-learn      # TF-IDF, modelos ML
gensim            # Word2Vec, FastText
sentence-transformers  # Embeddings de sentenças

# Dados
pandas            # DataFrames
numpy             # Arrays
beautifulsoup4    # Limpeza HTML

# Web
flask             # Deployment

# Visualização
matplotlib        # Gráficos
seaborn           # Heatmaps
```

---

## DATASETS UTILIZADOS

```python
# Fundamentos
'noticias_sinteticas.csv'  # Pequeno, didático

# Projetos
'B2W-Reviews01.csv'        # 129k reviews PT-BR
'sugestoes.txt'            # 1.5k sugestões IA
'FAQ_BB.json'              # 1.2k FAQs Banco Central
```

---

## ESTATÍSTICAS DO MÓDULO

**Documentação**:
- 12 arquivos markdown
- ~5.000 linhas de código
- ~120.000 palavras

**Código**:
- 9 notebooks
- 2 aplicações Flask
- 200+ snippets reutilizáveis

**Tempo**:
- Trilha completa: 3-4 semanas
- Trilha rápida: 1 semana

---

## TAGS DE BUSCA

`#nlp-classico` `#tfidf` `#word2vec` `#naive-bayes` `#svm` `#sentence-transformers` `#classificacao-texto` `#analise-sentimento` `#busca-semantica` `#sklearn` `#nltk` `#portuguese-nlp` `#flask` `#deployment`

---

**Versão**: 1.0  
**Compatibilidade**: Python 3.7+  
**Uso recomendado**: Aprendizado incremental, baseline rápido, produção leve
