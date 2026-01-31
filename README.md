# EAI_04 - NLP Clássico

Módulo completo de **Processamento de Linguagem Natural Clássico**, cobrindo fundamentos (pré-processamento, TF-IDF, embeddings), modelos tradicionais de ML e projetos deployados em produção.

---

## 🎯 Objetivo do Módulo

Dominar técnicas clássicas de NLP que são **base essencial** para:
- ✅ Entender sistemas modernos (BERT, GPT)
- ✅ Construir baselines rápidos e eficientes
- ✅ Deployar modelos leves em produção
- ✅ Resolver 80% dos problemas de NLP com ferramentas simples

**Por que estudar NLP Clássico em 2026?**
- Modelos pequenos e rápidos (deploy fácil)
- Menor custo computacional (CPU é suficiente)
- Mais interpretáveis (sabe por que o modelo decide)
- Ainda muito usados em produção real

---

## 📂 Estrutura do Módulo

```
EAI_04_NLP_Classico/
├── README.md (este arquivo)
├── AGENT_CONTEXT.md
│
├── Fundamentos/                    # 6 notebooks
│   ├── README.md
│   ├── AGENT_CONTEXT.md
│   ├── pre_processamento_texto.ipynb
│   ├── bow_tfidf.ipynb
│   ├── representacao_bow_tfidf.ipynb
│   ├── word_embeddings.ipynb
│   ├── pretrained_embeddings.ipynb
│   └── analise_sentimentos.ipynb
│
├── Modelos_Base/                   # 3 notebooks
│   ├── README.md
│   ├── AGENT_CONTEXT.md
│   ├── naive_bayes_sentimentos.ipynb
│   ├── classificacao_texto_svm.ipynb
│   └── comparativo_tfidf_vs_embeddings.ipynb
│
└── Projetos/                       # 2 aplicações
    ├── Analise_de_Feedback/
    │   ├── README.md
    │   ├── AGENT_CONTEXT.md
    │   ├── app.py (Flask)
    │   ├── notebook/
    │   ├── models/
    │   ├── templates/
    │   └── static/
    │
    └── Sistema_de_Busca_FAQs/
        ├── README.md
        ├── AGENT_CONTEXT.md
        ├── app.py (Flask)
        ├── notebook/
        ├── models/
        ├── templates/
        └── static/
```

**Total**: 12 arquivos de documentação + 9 notebooks + 2 projetos deployados

---

## 🗺️ Jornada de Aprendizado

### Trilha Completa (Recomendado)

#### Semana 1: Fundamentos (6 notebooks)
```
Dia 1-2: Pré-processamento e BoW/TF-IDF
  └─ pre_processamento_texto.ipynb
  └─ bow_tfidf.ipynb

Dia 3-4: TF-IDF Avançado e Aplicação
  └─ representacao_bow_tfidf.ipynb
  └─ analise_sentimentos.ipynb

Dia 5-6: Word Embeddings
  └─ word_embeddings.ipynb
  └─ pretrained_embeddings.ipynb
```

#### Semana 2: Modelos Base (3 notebooks)
```
Dia 1-2: Naive Bayes
  └─ naive_bayes_sentimentos.ipynb

Dia 3-4: SVM
  └─ classificacao_texto_svm.ipynb

Dia 5-6: Comparação
  └─ comparativo_tfidf_vs_embeddings.ipynb
```

#### Semana 3-4: Projetos (2 aplicações)
```
Semana 3: Análise de Feedback
  └─ Estudo do código
  └─ Execução local
  └─ Adaptação para seus dados

Semana 4: Sistema de Busca FAQs
  └─ Estudo do código
  └─ Execução local
  └─ Experimentação
```

---

### Trilha Rápida (1 semana)

Para quem já tem experiência com ML:

```
Dia 1: Fundamentos essenciais
  └─ bow_tfidf.ipynb
  └─ word_embeddings.ipynb

Dia 2-3: Modelos
  └─ naive_bayes_sentimentos.ipynb
  └─ classificacao_texto_svm.ipynb

Dia 4-5: Comparação e Projetos
  └─ comparativo_tfidf_vs_embeddings.ipynb
  └─ Executar os 2 projetos
```

---

## 📚 Conteúdo Detalhado

### 1️⃣ Fundamentos (Teoria + Prática)

#### 🔧 pre_processamento_texto.ipynb
- Limpeza de texto (lowercase, pontuação)
- Stopwords e remoção
- Stemming vs Lemmatization
- Pipeline completo reutilizável

#### 📊 bow_tfidf.ipynb
- Bag of Words (contagem)
- TF-IDF (importância)
- Comparação lado a lado
- Fórmulas matemáticas

#### 📈 representacao_bow_tfidf.ipynb
- N-gramas (uni, bi, tri)
- Parâmetros (max_df, min_df, max_features)
- Normalização L2
- Otimização de vocabulário

#### 🧠 word_embeddings.ipynb
- Word2Vec (CBOW vs Skip-gram)
- FastText (subpalavras)
- Treinamento do zero
- Aritmética semântica (rei - homem + mulher ≈ rainha)

#### 💎 pretrained_embeddings.ipynb
- NILC Word2Vec (português)
- GloVe (inglês)
- Como usar em modelos
- Vantagens vs treinar do zero

#### 🎯 analise_sentimentos.ipynb
- Pipeline end-to-end
- TF-IDF + Logistic Regression
- Classificação Positivo/Negativo
- Aplicação prática completa

---

### 2️⃣ Modelos Base (Templates Prontos)

#### 🚀 naive_bayes_sentimentos.ipynb
**Modelo**: Multinomial Naive Bayes  
**Performance**: ~85-88% accuracy  
**Quando usar**: Baseline rápido, poucos dados

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('nb', MultinomialNB(alpha=1.0))
])
```

#### ⚡ classificacao_texto_svm.ipynb
**Modelo**: LinearSVC  
**Performance**: ~89-92% accuracy  
**Quando usar**: Melhor performance, produção

```python
Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2)
    )),
    ('svm', LinearSVC(C=1.0))
])
```

#### 🔬 comparativo_tfidf_vs_embeddings.ipynb
**Experimento**: 3 abordagens comparadas  
**Resultado típico**:
- TF-IDF: ~87%
- Word2Vec próprio: ~85%
- Embeddings pré-treinados: ~90% ⭐

---

### 3️⃣ Projetos (Aplicações Reais)

#### 📊 Projeto 1: Análise de Feedback

**Problema**: Classificar feedbacks automaticamente  
**Solução**: 2 modelos especializados

**Modelo 1 - Sentimento**:
- Positivo vs Negativo
- Dataset: B2W-Reviews (129k)
- Accuracy: 95%

**Modelo 2 - Sugestão**:
- Contém sugestão de melhoria?
- Dataset: Sugestões IA (3k)
- Accuracy: 98%

**Deploy**: Flask web app  
**Diferencial**: Dupla classificação > modelo único

---

#### 🔍 Projeto 2: Sistema de Busca FAQs

**Problema**: Buscar FAQs por significado (não palavras exatas)  
**Solução**: Busca semântica com Sentence Transformers

**Tecnologia**:
- Modelo: distiluse-base-multilingual-cased-v1
- Método: Similaridade de cosseno
- Dataset: 1.172 FAQs do Banco Central

**Performance**:
- Top-1 Accuracy: ~75%
- Top-3 Accuracy: ~90%
- Velocidade: <1s

**Deploy**: Flask web app  
**Diferencial**: Entende sinônimos e contexto

---

## 📊 Comparação de Técnicas

### Representações de Texto

| Técnica | Dimensões | Tipo | Semântica | Quando Usar |
|---------|-----------|------|-----------|-------------|
| **BoW** | 10k-100k | Esparso | ❌ | Baseline rápido |
| **TF-IDF** | 10k-100k | Esparso | ❌ | Classificação, busca |
| **Word2Vec** | 100-300 | Denso | ✅ | Similaridade, clustering |
| **FastText** | 100-300 | Denso | ✅ | Palavras fora vocabulário |
| **Sentence Transformers** | 512-768 | Denso | ✅✅ | Busca semântica, Q&A |

### Modelos de ML

| Modelo | Accuracy | Treino | Predição | Interpretável |
|--------|----------|--------|----------|---------------|
| **Naive Bayes** | 85% | ⚡⚡⚡ | ⚡⚡⚡ | ⭐⭐⭐ |
| **Logistic Reg** | 87% | ⚡⚡ | ⚡⚡ | ⭐⭐⭐ |
| **LinearSVC** | 90% | ⚡ | ⚡⚡ | ⭐⭐ |

---

## 💻 Instalação

### Requisitos
```
Python 3.7+
8GB RAM (mínimo)
10GB espaço em disco
```

### Setup Completo
```bash
# 1. Clonar repositório
git clone https://github.com/usuario/EAI_04_NLP_Classico.git
cd EAI_04_NLP_Classico

# 2. Criar ambiente
conda create -n nlp_env python=3.9
conda activate nlp_env

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Download recursos NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"
```

### Dependências Principais
```txt
# NLP Clássico
nltk>=3.8
gensim>=4.3
sentence-transformers>=2.2

# ML
scikit-learn>=1.3

# Dados
pandas>=2.0
numpy>=1.24

# Web
flask>=2.3

# Utilidades
beautifulsoup4>=4.12
matplotlib>=3.7
seaborn>=0.12
```

---

## 🚀 Execução Rápida

### Notebooks
```bash
# Jupyter
jupyter notebook Fundamentos/

# Ou JupyterLab
jupyter lab
```

### Projetos

**Análise de Feedback**:
```bash
cd Projetos/Analise_de_Feedback
python app.py
# Acesse: http://localhost:5000
```

**Sistema de Busca FAQs**:
```bash
cd Projetos/Sistema_de_Busca_FAQs
python app.py
# Acesse: http://localhost:5000
```

---

## 🎯 Checklist de Conclusão

### Fundamentos
- [ ] Executei os 6 notebooks
- [ ] Entendo BoW vs TF-IDF vs Embeddings
- [ ] Sei quando usar cada representação
- [ ] Domino pré-processamento de texto

### Modelos
- [ ] Treinei Naive Bayes
- [ ] Treinei SVM
- [ ] Comparei diferentes abordagens
- [ ] Sei escolher modelo por tarefa

### Projetos
- [ ] Executei Análise de Feedback
- [ ] Executei Sistema de Busca FAQs
- [ ] Testei com inputs próprios
- [ ] Adaptei para meus dados

### Avançado
- [ ] Criei projeto próprio de NLP
- [ ] Fiz deploy em produção
- [ ] Integrei com sistema existente

---

## 📖 Recursos Complementares

### Cursos Online
- [Coursera: NLP Specialization](https://www.coursera.org/specializations/natural-language-processing)
- [Fast.ai: NLP](https://www.fast.ai/)
- [deeplearning.ai](https://www.deeplearning.ai/)

### Livros Recomendados
- "Speech and Language Processing" - Jurafsky & Martin
- "Natural Language Processing with Python" - Bird, Klein & Loper
- "Applied Text Analysis with Python" - Bengfort, Bilbro & Ojeda

### Bibliotecas
- [NLTK](https://www.nltk.org/) - Toolkit clássico
- [spaCy](https://spacy.io/) - NLP moderno
- [Gensim](https://radimrehurek.com/gensim/) - Word2Vec
- [Sentence Transformers](https://www.sbert.net/) - Embeddings

### Datasets em Português
- [B2W-Reviews](https://github.com/americanas-tech/b2w-reviews01)
- [IMDB-PT](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr)
- [TweetSentBR](https://bitbucket.org/HBrum/tweetsentbr)

---

## 🔮 Próximos Passos

Após completar EAI_04, você estará pronto para:

### EAI_05 - NLP Moderno
- Transformers (arquitetura)
- BERT, RoBERTa, GPT
- Fine-tuning
- Hugging Face Transformers

### EAI_06 - NLP Avançado
- Question Answering
- Named Entity Recognition
- Machine Translation
- Text Generation

### Projetos Avançados
- Chatbots conversacionais
- RAG (Retrieval Augmented Generation)
- Sistemas multilingues
- Análise de sentimento multimodal

---

## 🤝 Contribuindo

Contribuições são bem-vindas!

**Como contribuir**:
1. Fork o repositório
2. Crie branch (`git checkout -b feature/melhoria`)
3. Commit mudanças
4. Push para branch
5. Abra Pull Request

**Ideias**:
- Novos notebooks de exemplos
- Datasets adicionais
- Melhorias na documentação
- Correções de bugs
- Traduções

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)  
**LinkedIn**: [carlos-henrique-bamberg-marques](https://www.linkedin.com/in/carlos-henrique-bamberg-marques/)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja `LICENSE` para detalhes.

---

## 🙏 Agradecimentos

- Comunidade brasileira de NLP
- Autores de bibliotecas open-source (NLTK, spaCy, Gensim)
- Datasets públicos (B2W, BCB)
- Alunos e contribuidores

---

## 📊 Estatísticas do Módulo

**Conteúdo**:
- 6 notebooks fundamentais
- 3 notebooks de modelos
- 2 projetos deployados
- 12 arquivos de documentação
- ~5.000 linhas de código
- ~120.000 palavras de documentação

**Tempo estimado**: 3-4 semanas (dedicação parcial)

**Nível**: Iniciante a Intermediário

---

**💡 Lembre-se**: NLP Clássico é a base. Domine antes de partir para Transformers!

*Desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_04*
