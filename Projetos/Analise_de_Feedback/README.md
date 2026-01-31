# 📊 Análise de Feedback com NLP Clássico

Sistema de análise automática de feedbacks com **dupla classificação**: sentimento (positivo/negativo) e detecção de sugestões de melhoria usando TF-IDF e Logistic Regression.

---

## 🎯 Objetivo

Criar uma ferramenta prática para empresas analisarem feedbacks de clientes automaticamente, identificando:
1. **Sentimento**: Se o feedback é positivo ou negativo
2. **Sugestão**: Se contém uma sugestão de melhoria

**Resultado**: 2 modelos com >95% de accuracy e interface web Flask.

---

## 🧠 Como Funciona

O sistema processa cada feedback em **dois pipelines independentes**:

### Pipeline Geral
```
Feedback do Usuário
    ↓
┌───────────────┴───────────────┐
│                               │
Modelo 1: Sentimento    Modelo 2: Sugestão
(Positivo/Negativo)     (Sim/Não)
    ↓                           ↓
Confiança: 92%          Confiança: 98%
    ↓                           ↓
└───────────────┬───────────────┘
                ↓
    Resultado Combinado
```

### Diferencial: Dois Modelos Especializados

**Por que 2 modelos em vez de 1?**
- ✅ **Separação de conceitos**: Sentimento ≠ Sugestão
- ✅ **Melhor accuracy**: Modelos especializados > modelo genérico
- ✅ **Flexibilidade**: Pode usar apenas 1 modelo se necessário

---

## 🏗️ Arquitetura dos Modelos

### Modelo 1: Classificação de Sentimento

**Dataset**: B2W-Reviews01 (129k avaliações de produtos)
```python
# Mapeamento
Rating 1-2 → Negativo (0)
Rating 4-5 → Positivo (1)
Rating 3   → Ignorado (neutro)
```

**Pipeline**:
```python
Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),    # Uni + Bigramas
        max_features=50000
    )),
    ('clf', LogisticRegression(max_iter=1000))
])
```

**Performance**:
```
Accuracy: 95%

              precision    recall  f1-score
Negativo          0.93      0.91      0.92
Positivo          0.96      0.97      0.97
```

---

### Modelo 2: Detecção de Sugestão

**Dataset**: 
- **Sugestões** (1.506): Geradas por IA (classe 1)
- **Opiniões puras** (1.506): Filtradas do B2W (classe 0)

**Filtro de Palavras-chave** (removidas do B2W):
```python
keywords = [
    'sugiro', 'sugestão', 'poderia', 'deveria',
    'recomendo que', 'adicionar', 'melhorar',
    'implementar', 'faltou', 'seria bom se'
]
```

**Pipeline**: Idêntico ao Modelo 1

**Performance**:
```
Accuracy: 98%

              precision    recall  f1-score
Não-Sugestão      0.97      0.99      0.98
Sugestão          0.99      0.97      0.98
```

---

## 📊 Datasets Utilizados

### 1. B2W-Reviews01.csv

**Fonte**: https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets

**Características**:
- 129.098 avaliações de produtos
- Ratings: 1-5 estrelas
- Texto em português
- E-commerce brasileiro

**Uso**: 
- Treino do Modelo de Sentimento
- Base para não-sugestões (filtrada)

### 2. sugestoes.txt

**Fonte**: Gerado por múltiplas IAs

**Características**:
- 1.506 sugestões variadas
- Formato diverso (formal, informal)
- Domínios variados

**Uso**: Classe positiva do Modelo de Sugestão

---

## 🚀 Como Usar

### 1. Instalação

```bash
# Clonar repositório
git clone https://github.com/usuario/analise-feedback.git
cd analise-feedback

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinar Modelos

```bash
# Executar notebook de treinamento
jupyter notebook notebook/treinamento_modelos.ipynb

# Ou via Python
python scripts/train_models.py
```

**Modelos salvos em**: `models/`
- `sentiment_pipeline.pkl`
- `suggestion_pipeline.pkl`

### 3. Executar Aplicação Flask

```bash
python app.py
```

**Acesse**: http://localhost:5000

### 4. Usar Interface

1. Digite ou cole um feedback
2. Clique em **"Analisar"**
3. Veja resultado:
   - **Sentimento**: Positivo/Negativo + Confiança
   - **Sugestão**: Sim/Não + Confiança

---

## 📁 Estrutura do Projeto

```
Analise_de_Feedback/
├── app.py                      # 🌐 Backend Flask
├── requirements.txt            # 📦 Dependências
├── README.md                   # 📄 Este arquivo
├── AGENT_CONTEXT.md           # 🤖 Documentação técnica
│
├── data/
│   ├── B2W-Reviews01.csv      # Dataset de avaliações
│   └── sugestoes.txt          # Dataset de sugestões
│
├── models/                     # 💾 Modelos treinados
│   ├── sentiment_pipeline.pkl
│   └── suggestion_pipeline.pkl
│
├── notebook/
│   └── treinamento_modelos.ipynb  # 📓 Treinamento
│
├── static/
│   └── css/
│       └── style.css          # 🎨 Estilos
│
└── templates/                  # 🖼️ Interface web
    ├── index.html
    └── resultado.html
```

---

## 🌐 Aplicação Flask

### Backend (app.py)

```python
from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Carregar modelos
pipeline_sentimento = load('models/sentiment_pipeline.pkl')
pipeline_sugestao = load('models/suggestion_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '').strip()
    
    # Modelo 1: Sentimento
    sentiment = pipeline_sentimento.predict([message])[0]
    sentiment_conf = pipeline_sentimento.predict_proba([message])[0].max() * 100
    
    # Modelo 2: Sugestão
    is_suggestion = pipeline_sugestao.predict([message])[0]
    suggestion_conf = pipeline_sugestao.predict_proba([message])[0].max() * 100
    
    return render_template('resultado.html',
                         review=message,
                         sentiment_prediction='Positivo' if sentiment==1 else 'Negativo',
                         sentiment_confidence=f"{sentiment_conf:.2f}%",
                         is_suggestion='Sim' if is_suggestion==1 else 'Não',
                         suggestion_confidence=f"{suggestion_conf:.2f}%")
```

### Frontend

**index.html**: Formulário de entrada  
**resultado.html**: Exibição de resultados com confiança

---

## 📚 Tecnologias Utilizadas

| Categoria | Tecnologia | Uso |
|-----------|-----------|-----|
| **NLP** | scikit-learn | TF-IDF, Logistic Regression |
| **Dados** | Pandas, NumPy | Manipulação de datasets |
| **Web** | Flask | Backend |
| **Frontend** | HTML/CSS | Interface |
| **ML** | joblib | Salvar/carregar modelos |

---

## 📊 Exemplos de Uso

### Exemplo 1: Feedback Positivo com Sugestão

**Input**:
```
"Adorei o produto! A entrega foi rápida. 
Sugiro que vocês adicionem mais opções de cores."
```

**Output**:
```
Sentimento: Positivo (Confiança: 94.3%)
Sugestão:   Sim      (Confiança: 97.8%)
```

---

### Exemplo 2: Feedback Negativo sem Sugestão

**Input**:
```
"Produto de péssima qualidade. Não recomendo."
```

**Output**:
```
Sentimento: Negativo (Confiança: 98.2%)
Sugestão:   Não      (Confiança: 99.1%)
```

---

### Exemplo 3: Feedback Positivo sem Sugestão

**Input**:
```
"Excelente! Superou minhas expectativas."
```

**Output**:
```
Sentimento: Positivo (Confiança: 99.5%)
Sugestão:   Não      (Confiança: 98.7%)
```

---

### Exemplo 4: Sugestão com Sentimento Neutro

**Input**:
```
"Poderiam implementar um sistema de rastreamento em tempo real."
```

**Output**:
```
Sentimento: Positivo (Confiança: 62.3%)  ← Baixa confiança
Sugestão:   Sim      (Confiança: 99.2%)
```

---

## 🔍 Como os Modelos Decidem?

### TF-IDF Captura Palavras-chave

**Sentimento Positivo**:
- "adorei", "excelente", "recomendo", "superou", "rápido"

**Sentimento Negativo**:
- "péssimo", "horrível", "não recomendo", "pior", "demora"

**Sugestão**:
- "sugiro", "poderia", "deveria", "seria bom", "implementar"

### N-gramas Capturam Contexto

**Unigramas**: ["adorei", "produto"]  
**Bigramas**: ["adorei o", "o produto"]

**Vantagem**: Captura negações
- "não gostei" vs "gostei"
- "não recomendo" vs "recomendo"

---

## 📈 Performance e Limitações

### Quando Funciona Bem

- ✅ Feedbacks claros e diretos
- ✅ Linguagem formal ou semi-formal
- ✅ Textos em português brasileiro
- ✅ Sugestões explícitas ("sugiro", "poderia")

### Quando Pode Falhar

- ❌ Ironia ou sarcasmo
- ❌ Sugestões implícitas (sem palavras-chave)
- ❌ Textos muito curtos (< 5 palavras)
- ❌ Linguagem muito informal (gírias)

### Métricas Reais

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **Sentimento** | 95% | 0.94 | 0.94 | 0.94 |
| **Sugestão** | 98% | 0.98 | 0.98 | 0.98 |

---

## 🔮 Melhorias Futuras

### Modelos
- [ ] Adicionar classe "Neutro" no sentimento
- [ ] Detectar urgência na sugestão (alta/média/baixa)
- [ ] Classificar tipo de sugestão (produto, entrega, atendimento)
- [ ] Usar BERT para capturar contexto melhor

### Dados
- [ ] Expandir dataset de sugestões (10k+ exemplos)
- [ ] Adicionar validação humana (5-10% do dataset)
- [ ] Balancear melhor positivo/negativo
- [ ] Incluir dados de outros domínios (hotéis, restaurantes)

### Aplicação
- [ ] API REST para integração
- [ ] Upload de arquivo CSV em lote
- [ ] Dashboard com estatísticas
- [ ] Exportar resultados (Excel, PDF)
- [ ] Deploy em cloud (Heroku, Railway)

### Análise
- [ ] Explicabilidade (LIME, SHAP)
- [ ] Visualizar palavras mais importantes
- [ ] Clustering de feedbacks similares
- [ ] Tendências ao longo do tempo

---

## 🤝 Contribuindo

Contribuições são bem-vindas!

**Como contribuir**:
1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

**Ideias de contribuição**:
- Adicionar mais classes (urgência, categoria)
- Melhorar interface web
- Implementar API REST
- Adicionar testes automatizados
- Criar dashboard de estatísticas

---

## 📖 Recursos Adicionais

### Datasets Similares
- [IMDB Reviews](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Amazon Reviews](https://nijianmo.github.io/amazon/index.html)
- [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

### Papers
- [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Logistic Regression for Text](https://www.aclweb.org/anthology/P02-1053/)

### Ferramentas
- [spaCy](https://spacy.io/) - NLP moderno
- [NLTK](https://www.nltk.org/) - NLP clássico
- [Gensim](https://radimrehurek.com/gensim/) - Topic modeling

---

## 📝 Citação

Se usar este projeto, por favor cite:

```
@misc{analise_feedback_2026,
  author = {Carlos Henrique Bamberg Marques},
  title = {Análise de Feedback com Dupla Classificação NLP},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/usuario/analise-feedback}
}
```

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 🙏 Agradecimentos

- [B2W Digital](https://github.com/americanas-tech/b2w-reviews01) - Dataset de reviews
- [Kaggle](https://www.kaggle.com/) - Plataforma de datasets
- [scikit-learn](https://scikit-learn.org/) - Biblioteca de ML
- Comunidade de NLP brasileira

---

**💡 Dica**: Use este sistema como baseline. Para produção real, considere modelos mais robustos (BERT, RoBERTa).

*Projeto desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_04*
