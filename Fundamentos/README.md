# Fundamentos de NLP Clássico

## 📌 Sobre

Esta pasta contém **notebooks fundamentais** que explicam as técnicas clássicas de Processamento de Linguagem Natural (NLP), desde pré-processamento de texto até representações vetoriais (BoW, TF-IDF, Word Embeddings).

**Objetivo**: Fornecer base sólida em NLP clássico antes de partir para modelos modernos (Transformers, BERT).

---

## 🎯 Por Que NLP Clássico?

Mesmo com modelos modernos como BERT e GPT, técnicas clássicas são essenciais:
- ✅ **Baseline rápido**: TF-IDF + SVM funciona bem em muitos casos
- ✅ **Eficiência**: Menor custo computacional
- ✅ **Interpretabilidade**: Mais fácil entender o que o modelo aprendeu
- ✅ **Produção**: Modelos menores e mais rápidos para deploy

---

## 📂 Notebooks Disponíveis

### 1️⃣ **pre_processamento_texto.ipynb** (Fundação)

**Tópicos**:
- Limpeza de texto (lowercase, remoção de pontuação)
- Remoção de stopwords (palavras comuns)
- Tokenização (quebrar texto em palavras)
- Stemming vs Lemmatization
- Normalização (acentos, espaços)

**Técnicas**:
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Remover pontuação
texto = re.sub(r'[^\w\s]', '', texto)

# Remover stopwords
stopwords_pt = set(stopwords.words('portuguese'))
palavras = [p for p in palavras if p not in stopwords_pt]

# Stemming
stemmer = RSLPStemmer()
palavras_stem = [stemmer.stem(p) for p in palavras]
```

**Para Quem**: Todos - é a base de qualquer projeto NLP

**Duração**: ~20 minutos

---

### 2️⃣ **bow_tfidf.ipynb** (Representações Básicas)

**Tópicos**:

#### Bag of Words (BoW)
- Conta frequência de cada palavra
- Ignora ordem e contexto
- Vetor esparso (muitos zeros)

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(textos)

# Exemplo: "gato" aparece 3 vezes → valor = 3
```

#### TF-IDF (Term Frequency - Inverse Document Frequency)
- **TF**: Frequência do termo no documento
- **IDF**: Importância do termo no corpus
- Penaliza palavras muito comuns

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(textos)

# Palavras raras têm peso maior
# Palavras comuns têm peso menor
```

**Fórmulas**:
```
TF(t,d) = (número de vezes que t aparece em d) / (total de termos em d)

IDF(t) = log(N / df(t))
  onde N = total de documentos
       df(t) = documentos que contêm t

TF-IDF(t,d) = TF(t,d) × IDF(t)
```

**Comparação**:
| Aspecto | BoW | TF-IDF |
|---------|-----|--------|
| **Pesos** | Contagem simples | Importância relativa |
| **Palavras comuns** | Peso alto | Peso baixo |
| **Palavras raras** | Peso baixo | Peso alto |
| **Uso** | Baseline | Classificação, busca |

**Para Quem**: Iniciantes em NLP

**Duração**: ~30 minutos

---

### 3️⃣ **representacao_bow_tfidf.ipynb** (Aprofundamento)

**Tópicos**:
- Variações de BoW (n-gramas)
- Parâmetros do TF-IDF (max_df, min_df)
- Normalização L2
- Análise de dimensionalidade

**N-gramas**:
```python
# Unigramas: ["bom", "filme"]
# Bigramas: ["bom filme"]
# Trigramas: ["muito bom filme"]

vectorizer = TfidfVectorizer(ngram_range=(1,2))  # Uni + Bigramas
```

**Filtros**:
```python
vectorizer = TfidfVectorizer(
    max_df=0.8,  # Ignora palavras em >80% dos docs
    min_df=2,    # Ignora palavras em <2 docs
    max_features=1000  # Top 1000 features
)
```

**Para Quem**: Intermediário

**Duração**: ~40 minutos

---

### 4️⃣ **word_embeddings.ipynb** (Representações Densas)

**Tópicos**:

#### Word2Vec
- Vetores densos (ex: 100 dimensões)
- Captura semântica: "rei" - "homem" + "mulher" ≈ "rainha"
- 2 arquiteturas: CBOW e Skip-gram

```python
from gensim.models import Word2Vec

# Treinar Word2Vec
model = Word2Vec(
    sentences=corpus_tokenizado,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# Obter vetor de uma palavra
vetor_rei = model.wv['rei']

# Palavras similares
similares = model.wv.most_similar('rei', topn=5)
```

#### FastText
- Extensão do Word2Vec
- Usa subpalavras (character n-grams)
- Funciona com palavras fora do vocabulário

```python
from gensim.models import FastText

model = FastText(
    sentences=corpus_tokenizado,
    vector_size=100,
    window=5,
    min_count=2
)

# Funciona mesmo para palavras novas!
vetor_palavra_nova = model.wv['palavrainexistente123']
```

**BoW/TF-IDF vs Word Embeddings**:
| Aspecto | BoW/TF-IDF | Word Embeddings |
|---------|------------|-----------------|
| **Dimensionalidade** | Alta (10k-100k) | Baixa (50-300) |
| **Esparsidade** | Esparso | Denso |
| **Semântica** | Não captura | Captura |
| **OOV** | Ignora | FastText funciona |

**Para Quem**: Intermediário a avançado

**Duração**: ~1 hora

---

### 5️⃣ **pretrained_embeddings.ipynb** (Embeddings Pré-treinados)

**Tópicos**:
- Carregar Word2Vec pré-treinado (Google, NILC)
- GloVe embeddings
- Como usar em modelos

**Word2Vec NILC (Português)**:
```python
from gensim.models import KeyedVectors

# Carregar modelo pré-treinado
model = KeyedVectors.load_word2vec_format(
    'skip_s300.txt',
    binary=False
)

# Usar vetores
vetor = model['computador']
similares = model.most_similar('computador', topn=10)
```

**Vantagens**:
- ✅ Não precisa treinar (economia de tempo)
- ✅ Treinado em corpus gigante (melhor qualidade)
- ✅ Funciona bem com poucos dados

**Fontes**:
- [NILC Word2Vec](http://nilc.icmc.usp.br/embeddings)
- [GloVe](https://nlp.stanford.edu/projects/glove/)
- [FastText Facebook](https://fasttext.cc/docs/en/crawl-vectors.html)

**Para Quem**: Quem quer pular treino de embeddings

**Duração**: ~30 minutos

---

### 6️⃣ **analise_sentimentos.ipynb** (Aplicação Prática)

**Tópicos**:
- Classificação de sentimento (positivo/negativo)
- Dataset de reviews
- Pipeline completo: Pré-processamento → TF-IDF → Modelo

**Pipeline**:
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Pipeline end-to-end
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])

# Treinar
pipeline.fit(X_train, y_train)

# Prever
sentimento = pipeline.predict(["Adorei o filme!"])
# Output: 'positivo'
```

**Dataset Típico**:
```
Texto: "Filme excelente, recomendo!"
Sentimento: positivo

Texto: "Péssimo, perdi meu tempo"
Sentimento: negativo
```

**Métricas**:
- Accuracy: ~85-90%
- Precision/Recall por classe

**Para Quem**: Todos - aplicação prática imediata

**Duração**: ~45 minutos

---

## 🗺️ Ordem de Estudo Recomendada

### Iniciante (Nunca viu NLP)
```
1. pre_processamento_texto.ipynb      (base essencial)
2. bow_tfidf.ipynb                     (primeira representação)
3. analise_sentimentos.ipynb           (aplicação prática)
4. word_embeddings.ipynb               (representação avançada)
5. pretrained_embeddings.ipynb         (usar embeddings prontos)
6. representacao_bow_tfidf.ipynb       (aprofundamento)
```

### Intermediário (Já conhece ML)
```
1. pre_processamento_texto.ipynb      (revisão rápida)
2. bow_tfidf.ipynb                     (conceitos)
3. word_embeddings.ipynb               (foco aqui)
4. analise_sentimentos.ipynb           (aplicação)
```

### Avançado (Revisão Rápida)
```
1. word_embeddings.ipynb               (conceitos chave)
2. pretrained_embeddings.ipynb         (uso prático)
3. Pular para Modelos_Base/
```

---

## 📊 Comparação de Técnicas

### Quando Usar Cada Representação?

| Técnica | Quando Usar | Vantagens | Desvantagens |
|---------|-------------|-----------|--------------|
| **BoW** | Baseline rápido, poucos dados | Simples, rápido | Ignora ordem, sem semântica |
| **TF-IDF** | Classificação, busca textual | Filtra palavras comuns | Ainda sem semântica |
| **Word2Vec** | Semântica importa, corpus médio | Captura relações | Precisa treinar |
| **FastText** | Palavras fora vocabulário | Funciona com OOV | Mais lento |
| **Pré-treinados** | Poucos dados, produção | Qualidade alta, rápido | Tamanho do arquivo |

---

## 🔑 Conceitos-Chave

### Pré-processamento

**Stopwords**: Palavras muito comuns sem valor semântico
```python
# Português
stopwords = ['a', 'o', 'de', 'da', 'em', 'para', 'com', ...]
```

**Stemming vs Lemmatization**:
```python
# Stemming (regras heurísticas)
correr → corr
corrida → corr
correndo → corr

# Lemmatization (análise linguística)
correr → correr
corrida → corrida
correndo → correr
```

### Vetorização

**Esparso vs Denso**:
```python
# BoW/TF-IDF: Esparso (muitos zeros)
[0, 0, 3, 0, 0, 1, 0, 0, ..., 0]  # 10.000 dimensões

# Word2Vec: Denso
[0.23, -0.45, 0.67, ..., 0.12]  # 100 dimensões
```

---

## 💻 Código Base Reutilizável

### Pipeline Completo de Pré-processamento

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

nltk.download('stopwords')
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
    
    # 3. Remover números
    texto = re.sub(r'\d+', '', texto)
    
    # 4. Tokenizar
    palavras = texto.split()
    
    # 5. Remover stopwords
    palavras = [p for p in palavras if p not in stopwords_pt]
    
    # 6. Stemming (opcional)
    palavras = [stemmer.stem(p) for p in palavras]
    
    # 7. Juntar
    return ' '.join(palavras)

# Uso
texto_limpo = preprocessar_texto("O filme foi excelente!")
```

### Pipeline Scikit-learn Completo

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Dados
X = ["texto 1", "texto 2", ...]
y = [0, 1, ...]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pipeline
pipeline = Pipeline([
    ('preprocessamento', FunctionTransformer(lambda x: [preprocessar_texto(t) for t in x])),
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LinearSVC())
])

# Treinar
pipeline.fit(X_train, y_train)

# Avaliar
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

---

## 🎯 Checklist de Aprendizado

### Conceitos Fundamentais
- [ ] Entendo diferença entre BoW, TF-IDF e Word Embeddings
- [ ] Sei quando usar stemming vs lemmatization
- [ ] Compreendo stopwords e por que removê-las
- [ ] Entendo vetores esparsos vs densos

### Técnicas
- [ ] Sei aplicar TF-IDF
- [ ] Sei treinar Word2Vec
- [ ] Sei usar embeddings pré-treinados
- [ ] Sei criar pipeline completo

### Prática
- [ ] Executei todos os 6 notebooks
- [ ] Apliquei em um dataset próprio
- [ ] Comparei BoW vs TF-IDF vs Word2Vec

---

## 📚 Recursos Complementares

### Bibliotecas
- [NLTK](https://www.nltk.org/) - Ferramentas NLP clássicas
- [spaCy](https://spacy.io/) - NLP moderno e rápido
- [Gensim](https://radimrehurek.com/gensim/) - Word2Vec, FastText

### Datasets
- [IMDB Reviews](http://ai.stanford.edu/~amaas/data/sentiment/) - Análise de sentimento
- [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) - Classificação de texto
- [B2W-Reviews](https://github.com/americanas-tech/b2w-reviews01) - Reviews em português

### Cursos
- [Coursera NLP Specialization](https://www.coursera.org/specializations/natural-language-processing)
- [Fast.ai NLP](https://www.fast.ai/)

---

## 🔧 Troubleshooting

### Problema: "Resource stopwords not found"
**Solução**:
```python
import nltk
nltk.download('stopwords')
nltk.download('rslp')  # Para stemmer português
```

### Problema: Vetores muito grandes (OOM)
**Solução**:
```python
# Limitar features
vectorizer = TfidfVectorizer(max_features=5000)

# Ou usar vetores esparsos
from scipy.sparse import save_npz
save_npz('vetores.npz', X_sparse)
```

### Problema: Word2Vec lento
**Solução**:
```python
# Usar workers
model = Word2Vec(sentences, workers=4)

# Ou usar embeddings pré-treinados
```

---

## 💡 Dicas de Estudo

1. **Execute célula por célula**
   - Não apenas leia, execute e observe

2. **Teste com seus textos**
   - Aplique em tweets, comentários, artigos

3. **Compare resultados**
   - BoW vs TF-IDF: Qual funciona melhor?

4. **Visualize embeddings**
   - Use t-SNE para ver agrupamentos

5. **Construa vocabulário**
   - Anote termos técnicos (corpus, token, stem)

---

## 🚀 Próximos Passos

Após dominar os fundamentos:

1. **Ir para Modelos_Base/** - Modelos prontos (SVM, Naive Bayes)
2. **Ir para Projetos/** - Aplicações deployadas
3. **Explorar NLP Moderno** - Transformers, BERT (EAI_05)

---

**Lembre-se**: NLP clássico ainda é MUITO usado em produção. Domine antes de partir para modelos complexos!

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_04*
