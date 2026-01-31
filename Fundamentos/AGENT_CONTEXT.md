# AGENT_CONTEXT.md - Fundamentos de NLP Clássico

> **Propósito**: Contexto técnico dos 6 notebooks fundamentais  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Notebooks educacionais conceituais

## RESUMO EXECUTIVO

**Objetivo**: Construir base sólida em técnicas clássicas de NLP  
**Notebooks**: 6 notebooks progressivos  
**Técnicas**: Pré-processamento, BoW, TF-IDF, Word2Vec, FastText, Embeddings pré-treinados  
**Duração total**: ~4-6 horas  
**Pré-requisitos**: Python básico, pandas  
**Resultado esperado**: Domínio de representações de texto

---

## NOTEBOOK 1: pre_processamento_texto.ipynb

### Objetivo Pedagógico
Ensinar pipeline completo de limpeza de texto antes de vetorização.

### Técnicas Implementadas

#### 1. Lowercase
```python
texto = texto.lower()
# "Olá Mundo!" → "olá mundo!"
```

**Por quê?**: "Gato" ≠ "gato" para máquina

#### 2. Remoção de Pontuação
```python
import re
texto = re.sub(r'[^\w\s]', '', texto)
# "olá, mundo!" → "olá mundo"
```

**Por quê?**: Pontuação geralmente não carrega significado

#### 3. Tokenização
```python
palavras = texto.split()
# "olá mundo" → ["olá", "mundo"]
```

**Por quê?**: Separar texto em unidades menores

#### 4. Remoção de Stopwords
```python
from nltk.corpus import stopwords

stopwords_pt = set(stopwords.words('portuguese'))
palavras_filtradas = [p for p in palavras if p not in stopwords_pt]

# ["o", "gato", "é", "preto"] → ["gato", "preto"]
```

**Stopwords comuns PT**:
```python
['a', 'o', 'de', 'da', 'em', 'para', 'com', 'que', 'não', 'mais', ...]
```

**Por quê?**: Palavras muito comuns não discriminam documentos

#### 5. Stemming
```python
from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()
palavras_stem = [stemmer.stem(p) for p in palavras]

# ["correndo", "correr", "corrida"] → ["corr", "corr", "corr"]
```

**Algoritmo**: RSLP (Removedor de Sufixos da Língua Portuguesa)

**Vantagens**:
- Rápido
- Reduz vocabulário

**Desvantagens**:
- Pode gerar palavras inexistentes ("corr")
- Regras heurísticas (não entende contexto)

#### 6. Lemmatization (conceito)
```python
# Não muito usado em português por falta de bibliotecas robustas
# spaCy tem lemmatizer PT, mas é mais lento

# Exemplo conceitual:
# ["correndo", "correr", "corrida"] → ["correr", "correr", "corrida"]
```

**Vantagens**:
- Palavras reais
- Contexto gramatical

**Desvantagens**:
- Mais lento
- Requer dicionário

### Pipeline Completo
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

stopwords_pt = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def preprocessar_texto(texto):
    """
    Pipeline completo de pré-processamento
    """
    # 1. Lowercase
    texto = texto.lower()
    
    # 2. Remover pontuação
    texto = re.sub(r'[^\w\s]', '', texto)
    
    # 3. Remover números (opcional)
    texto = re.sub(r'\d+', '', texto)
    
    # 4. Tokenizar
    palavras = texto.split()
    
    # 5. Remover stopwords
    palavras = [p for p in palavras if p not in stopwords_pt]
    
    # 6. Stemming
    palavras = [stemmer.stem(p) for p in palavras]
    
    # 7. Juntar
    return ' '.join(palavras)

# Uso
texto_limpo = preprocessar_texto("O filme foi excelente!")
# Output: "film excel"
```

### Quando Aplicar Cada Etapa

| Etapa | Sempre | Opcional | Nunca |
|-------|--------|----------|-------|
| Lowercase | ✅ | | |
| Remover pontuação | ✅ | | |
| Remover números | | ✅ | |
| Stopwords | ✅ | | |
| Stemming | | ✅ | |
| Lemmatization | | ✅ | |

**Regra geral**: Teste com e sem cada etapa, veja o que funciona melhor.

---

## NOTEBOOK 2: bow_tfidf.ipynb

### Objetivo Pedagógico
Introduzir primeiras representações vetoriais de texto.

### Bag of Words (BoW)

#### Conceito
Representar documento por frequência de palavras, ignorando ordem.

#### Implementação
```python
from sklearn.feature_extraction.text import CountVectorizer

textos = [
    "gato preto",
    "cachorro preto",
    "gato branco"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

print(vectorizer.get_feature_names_out())
# ['branco', 'cachorro', 'gato', 'preto']

print(X.toarray())
# [[0 0 1 1]   ← "gato preto"
#  [0 1 0 1]   ← "cachorro preto"
#  [1 0 1 0]]  ← "gato branco"
```

#### Características
- **Esparso**: Muitos zeros
- **Alta dimensionalidade**: Vocabulário completo
- **Sem semântica**: "gato" e "felino" são diferentes
- **Sem ordem**: "não gostei" = "gostei não"

### TF-IDF (Term Frequency - Inverse Document Frequency)

#### Conceito
Dar peso maior para palavras raras e importantes.

#### Fórmulas
```
TF(t,d) = (# vezes que t aparece em d) / (# total de palavras em d)

IDF(t) = log(N / df(t))
onde:
  N = total de documentos
  df(t) = documentos que contêm t

TF-IDF(t,d) = TF(t,d) × IDF(t)
```

#### Implementação
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

print(X.toarray())
# Valores normalizados entre 0 e 1
# Palavras raras têm peso maior
```

#### Exemplo Detalhado
```python
textos = [
    "gato preto",           # Doc 0
    "cachorro preto",       # Doc 1
    "gato branco"           # Doc 2
]

# "preto" aparece em 2/3 docs → peso menor
# "gato" aparece em 2/3 docs → peso menor
# "cachorro" aparece em 1/3 docs → peso MAIOR
# "branco" aparece em 1/3 docs → peso MAIOR
```

### BoW vs TF-IDF

| Aspecto | BoW | TF-IDF |
|---------|-----|--------|
| **Valores** | Inteiros (contagem) | Float (0-1) |
| **Palavras comuns** | Peso alto | Peso baixo |
| **Palavras raras** | Peso baixo | Peso alto |
| **Normalização** | Não | Sim (L2) |
| **Uso** | Baseline | Classificação, busca |

### Código Comparativo
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# BoW
bow = CountVectorizer()
X_bow = bow.fit_transform(textos)

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(textos)

# Comparar
print("BoW:", X_bow[0].toarray())
print("TF-IDF:", X_tfidf[0].toarray())
```

---

## NOTEBOOK 3: representacao_bow_tfidf.ipynb

### Objetivo Pedagógico
Aprofundar em parâmetros e variações de BoW/TF-IDF.

### N-gramas

#### Conceito
Sequências de N palavras consecutivas.

```python
texto = "não gostei do filme"

# Unigramas (1-gram)
["não", "gostei", "do", "filme"]

# Bigramas (2-gram)
["não gostei", "gostei do", "do filme"]

# Trigramas (3-gram)
["não gostei do", "gostei do filme"]
```

#### Implementação
```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# Usa unigramas E bigramas

X = vectorizer.fit_transform(textos)
```

#### Por Que N-gramas?
```python
# Problema: Negação
"gostei" vs "não gostei"

# Com unigramas:
["não", "gostei"] → Ambos presentes (confuso)

# Com bigramas:
["não gostei"] → Claro que é negativo!
```

### Parâmetros Importantes

#### max_df (maximum document frequency)
```python
vectorizer = TfidfVectorizer(max_df=0.8)
# Ignora palavras que aparecem em >80% dos documentos
```

**Uso**: Remover palavras muito comuns (ex: "é", "a")

#### min_df (minimum document frequency)
```python
vectorizer = TfidfVectorizer(min_df=2)
# Ignora palavras que aparecem em <2 documentos
```

**Uso**: Remover palavras raríssimas (typos, nomes próprios)

#### max_features
```python
vectorizer = TfidfVectorizer(max_features=10000)
# Usa apenas top 10k palavras mais importantes
```

**Uso**: Reduzir dimensionalidade, evitar overfit

#### Exemplo Completo
```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # Uni + Bigramas
    max_df=0.8,              # Ignora >80% docs
    min_df=2,                # Ignora <2 docs
    max_features=10000,      # Top 10k
    sublinear_tf=True,       # log(TF)
    strip_accents='unicode'  # Remove acentos
)
```

### Normalização L2
```python
# TF-IDF aplica normalização L2 automaticamente
# Cada documento vira vetor unitário (||v|| = 1)

# Fórmula:
# v_normalized = v / ||v||
```

**Por quê?**: Documentos longos vs curtos ficam comparáveis

---

## NOTEBOOK 4: word_embeddings.ipynb

### Objetivo Pedagógico
Introduzir embeddings densos que capturam semântica.

### Word2Vec

#### Conceito
Representar palavras como vetores densos onde palavras similares ficam próximas.

```python
# Espaço vetorial (simplificado 2D)
"rei"    → [0.8, 0.3]
"rainha" → [0.7, 0.4]
"homem"  → [0.6, 0.1]
"mulher" → [0.5, 0.2]

# Relações:
rei - homem + mulher ≈ rainha
```

#### Arquiteturas

**CBOW (Continuous Bag of Words)**:
```
Contexto → Palavra Alvo

"O ___ está na mesa"
[O, está, na, mesa] → gato
```

**Skip-gram**:
```
Palavra Alvo → Contexto

gato → [O, está, na, mesa]
```

**Quando usar**:
- CBOW: Corpus grande, mais rápido
- Skip-gram: Corpus pequeno, melhor para palavras raras

#### Implementação
```python
from gensim.models import Word2Vec

# Preparar corpus
sentences = [
    ["o", "gato", "está", "na", "mesa"],
    ["o", "cachorro", "está", "no", "jardim"],
    ...
]

# Treinar
model = Word2Vec(
    sentences=sentences,
    vector_size=100,    # Dimensões do vetor
    window=5,           # Janela de contexto
    min_count=2,        # Ignora palavras com <2 ocorrências
    workers=4,          # Threads paralelas
    sg=0                # 0=CBOW, 1=Skip-gram
)

# Usar
vetor_gato = model.wv['gato']
# array de 100 dimensões

# Similaridade
similares = model.wv.most_similar('gato', topn=5)
# [('felino', 0.89), ('cachorro', 0.75), ...]
```

### FastText

#### Diferença do Word2Vec
```
Word2Vec:  "correndo" = vetor único
FastText:  "correndo" = soma de subpalavras
           ["corr", "orre", "rren", ..., "correndo"]
```

#### Vantagem
```python
# Palavra fora do vocabulário (OOV)
model_w2v.wv['palavrainexistente123']  # KeyError!

model_ft.wv['palavrainexistente123']   # Funciona!
# Cria vetor baseado em subpalavras
```

#### Implementação
```python
from gensim.models import FastText

model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)
```

### Quando Usar Word2Vec vs TF-IDF

| Aspecto | TF-IDF | Word2Vec |
|---------|--------|----------|
| **Dimensões** | 10k-100k | 50-300 |
| **Esparsidade** | Esparso | Denso |
| **Semântica** | ❌ | ✅ |
| **Treino** | Não precisa | Precisa corpus |
| **OOV** | Ignora | Problema (FastText resolve) |
| **Uso** | Classificação | Similaridade, clustering |

---

## NOTEBOOK 5: pretrained_embeddings.ipynb

### Objetivo Pedagógico
Usar embeddings pré-treinados sem necessidade de treinar.

### Embeddings Disponíveis

#### NILC Word2Vec (Português)
```python
from gensim.models import KeyedVectors

# Baixar de: http://nilc.icmc.usp.br/embeddings
# Modelos: skip_s300.txt, cbow_s300.txt

model = KeyedVectors.load_word2vec_format(
    'skip_s300.txt',
    binary=False
)

# Usar
vetor = model['computador']
similares = model.most_similar('computador', topn=10)
```

#### GloVe (Inglês)
```python
# Baixar de: https://nlp.stanford.edu/projects/glove/

model = KeyedVectors.load_word2vec_format(
    'glove.6B.100d.txt',
    binary=False,
    no_header=True
)
```

### Usar Embeddings em Modelos

#### Estratégia: Média dos Vetores
```python
import numpy as np

def document_vector(text, model):
    """
    Representa documento como média dos vetores das palavras
    """
    words = text.split()
    word_vectors = []
    
    for word in words:
        if word in model:
            word_vectors.append(model[word])
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)

# Uso
doc_vec = document_vector("o gato está na mesa", model)
# Array de 300 dimensões
```

#### Classificação com Embeddings
```python
from sklearn.linear_model import LogisticRegression

# Vetorizar documentos
X_train_vec = [document_vector(doc, model) for doc in X_train]
X_test_vec = [document_vector(doc, model) for doc in X_test]

# Treinar
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# Avaliar
accuracy = clf.score(X_test_vec, y_test)
```

### Vantagens dos Pré-treinados

- ✅ Não precisa treinar (economia de tempo)
- ✅ Treinado em bilhões de palavras (melhor qualidade)
- ✅ Funciona bem com poucos dados
- ❌ Tamanho grande (~1-5 GB)
- ❌ Domínio geral (não específico)

---

## NOTEBOOK 6: analise_sentimentos.ipynb

### Objetivo Pedagógico
Aplicação prática end-to-end combinando tudo.

### Pipeline Completo

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Dados
textos = ["Adorei o filme!", "Péssimo produto", ...]
sentimentos = ["positivo", "negativo", ...]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    textos, sentimentos, test_size=0.2, random_state=42
)

# 3. Pipeline (pré-processamento integrado)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])

# 4. Treinar
pipeline.fit(X_train, y_train)

# 5. Avaliar
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

# 6. Predição
novo_texto = "Filme maravilhoso!"
predicao = pipeline.predict([novo_texto])
print(f"Sentimento: {predicao[0]}")
```

### Performance Esperada
```
Accuracy: ~85-90%

              precision    recall  f1-score
positivo         0.87      0.85      0.86
negativo         0.84      0.86      0.85
```

### Pré-processamento Customizado
```python
from sklearn.base import TransformerMixin

class TextPreprocessor(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [preprocessar_texto(text) for text in X]

# Pipeline com pré-processamento
pipeline = Pipeline([
    ('preproc', TextPreprocessor()),
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
```

---

## CONCEITOS-CHAVE CONSOLIDADOS

### Representações de Texto

```python
# 1. BoW: Contagem
"gato gato cachorro" → [2, 1]  # [gato, cachorro]

# 2. TF-IDF: Importância
"gato gato cachorro" → [0.6, 0.8]  # cachorro mais raro

# 3. Word2Vec: Semântica
"gato" → [0.23, -0.45, 0.67, ...]  # 100D

# 4. Document Vector: Agregação
"gato cachorro" → mean([vec_gato, vec_cachorro])
```

### Quando Usar Cada Técnica

**BoW/TF-IDF**:
- Classificação de texto
- Busca por relevância
- Baseline rápido

**Word2Vec/FastText**:
- Similaridade semântica
- Clustering
- Features para modelos

**Embeddings Pré-treinados**:
- Poucos dados
- Domínio geral
- Qualidade alta necessária

---

## CHECKLIST DE CONCLUSÃO

- [ ] Entendo pré-processamento completo
- [ ] Sei diferença entre BoW e TF-IDF
- [ ] Sei quando usar n-gramas
- [ ] Treinei Word2Vec
- [ ] Usei embeddings pré-treinados
- [ ] Criei pipeline completo de análise de sentimento

---

## TAGS DE BUSCA

`#nlp-fundamentos` `#preprocessamento` `#bow` `#tfidf` `#word2vec` `#fasttext` `#embeddings` `#analise-sentimento` `#sklearn` `#gensim` `#nltk`

---

**Versão**: 1.0  
**Compatibilidade**: Python 3.7+  
**Uso recomendado**: Aprendizado sequencial, base para NLP avançado
