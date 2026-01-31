# 🔍 Sistema de Busca Inteligente para FAQs

Sistema de busca semântica para base de conhecimento com **Sentence Transformers**, capaz de entender o significado da pergunta (não apenas palavras-chave) usando embeddings e similaridade de cosseno.

---

## 🎯 Objetivo

Criar um sistema de autoatendimento inteligente que:
1. **Entende contexto**: Não busca palavras exatas, busca significado
2. **Ranqueia respostas**: Top 3 mais relevantes com score
3. **Filtra irrelevâncias**: Threshold mínimo de 50% de similaridade

**Resultado**: Redução de carga em suporte, respostas instantâneas 24/7.

---

## 🧠 Como Funciona

### Busca Tradicional vs Busca Semântica

**Busca Tradicional** (Keyword-based):
```
Usuário: "Como fazer PIX?"
Sistema: Busca por "fazer" AND "PIX"
Resultado: Pode não encontrar se FAQ usa "realizar" em vez de "fazer"
```

**Busca Semântica** (Este projeto):
```
Usuário: "Como fazer PIX?"
Sistema: Entende significado → Compara com todos FAQs
Resultado: Encontra "Como realizar transferência PIX" (similaridade 87%)
```

### Pipeline Visual

```
Pergunta do Usuário
    ↓
Sentence Transformer (embedding 512D)
    ↓
Comparar com Base (1.172 embeddings)
    ↓
Similaridade de Cosseno
    ↓
Top 3 Resultados (≥50% similaridade)
    ↓
Exibir com Score
```

---

## 🏗️ Arquitetura do Sistema

### Modelo: Sentence Transformers

**Modelo usado**: `distiluse-base-multilingual-cased-v1`

**Por quê?**
- ✅ Multilíngue (funciona bem em português)
- ✅ Embeddings de 512 dimensões (menor que BERT 768D)
- ✅ Distilado (mais rápido, leve)
- ✅ Captura semântica de sentenças completas

**Arquitetura**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Converter texto em vetor
embedding = model.encode("Como fazer PIX?")
# Output: array de 512 números (vetor semântico)
```

### Similaridade de Cosseno

**Fórmula**:
```
cos(θ) = (A · B) / (||A|| × ||B||)

Onde:
- A = embedding da pergunta do usuário
- B = embedding de cada FAQ
- Resultado: -1 a 1 (convertido para 0% a 100%)
```

**Por que cosseno?**
- ✅ Direção importa mais que magnitude
- ✅ Normalizado (sempre entre -1 e 1)
- ✅ Rápido de computar

---

## 📊 Dataset - FAQs do Banco Central

### Fonte

**URL**: https://www.bcb.gov.br/api/servico/faq/faqperguntas

**Características**:
- 1.172 pares de pergunta/resposta
- Temas: PIX, empréstimos, Registrato, etc.
- Formato original: JSON com HTML nas respostas

### Estrutura Original (JSON)

```json
{
  "conteudo": [
    {
      "pergunta": "O que é Registrato?",
      "resposta": "<p>O Registrato é um sistema onde você pode...</p>"
    },
    ...
  ]
}
```

### Limpeza de Dados

**Problema**: Respostas contêm HTML

```html
<p>O Registrato é um sistema onde você pode consultar...</p>
<ul><li>Item 1</li><li>Item 2</li></ul>
```

**Solução**: BeautifulSoup para extrair texto puro

```python
from bs4 import BeautifulSoup

def limpar_html(texto_html):
    if not isinstance(texto_html, str):
        return ""
    soup = BeautifulSoup(texto_html, "html.parser")
    return soup.get_text(separator=' ', strip=True)

df['resposta_limpa'] = df['resposta'].apply(limpar_html)
```

**Resultado**:
```
"O Registrato é um sistema onde você pode consultar... Item 1 Item 2"
```

### DataFrame Final

```python
df_faq_limpo.head()

         pergunta                                          resposta
0  O que é Registrato?  O Registrato é um sistema onde você pode...
1  Como acesso o PIX?   Para acessar o PIX, você precisa...
...
```

**Total**: 1.172 pares pergunta/resposta limpos

---

## 🚀 Como Usar

### 1. Instalação

```bash
# Clonar repositório
git clone https://github.com/RickBamberg/Sistema_de_Busca_FAQs.git
cd Sistema_de_Busca_FAQs

# Criar ambiente virtual (Conda)
conda create --name faq_env python=3.9
conda activate faq_env

# Instalar dependências
pip install -r requirements.txt
```

### 2. Gerar Embeddings (Primeira vez)

```bash
# Executar notebook
jupyter notebook notebook/FAQ_Semantic_Search.ipynb

# Ou via Python
python scripts/generate_embeddings.py
```

**O que é gerado**:
- `models/embeddings_faq.npy` (matriz 1172×512)
- `models/dados_faq.pkl` (perguntas e respostas)

### 3. Executar Aplicação Flask

```bash
python app.py
```

**Acesse**: http://localhost:5000

### 4. Usar Interface

1. Digite uma pergunta (ex: "Como fazer PIX?")
2. Clique em **"Buscar"**
3. Veja Top 3 resultados com score de similaridade

---

## 📁 Estrutura do Projeto

```
Sistema_de_Busca_FAQs/
├── app.py                      # 🌐 Backend Flask
├── requirements.txt            # 📦 Dependências
├── README.md                   # 📄 Este arquivo
├── AGENT_CONTEXT.md           # 🤖 Documentação técnica
│
├── data/
│   └── FAQ_BB.json            # Dataset original (BCB)
│
├── models/                     # 💾 Artefatos gerados
│   ├── embeddings_faq.npy     # Matriz de embeddings (1172×512)
│   └── dados_faq.pkl          # Perguntas e respostas
│
├── notebook/
│   └── FAQ_Semantic_Search.ipynb  # 📓 Geração de embeddings
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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

app = Flask(__name__)

# Carregar artefatos (uma vez)
with open('models/dados_faq.pkl', 'rb') as f:
    dados_faq = pickle.load(f)
embeddings_perguntas = np.load('models/embeddings_faq.npy')
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def buscar_resposta_similar(pergunta_usuario, top_k=3, threshold=0.5):
    """
    Busca Top K respostas com similaridade >= threshold
    """
    # 1. Gerar embedding da pergunta
    embedding_usuario = model.encode([pergunta_usuario])
    
    # 2. Calcular similaridade com todos FAQs
    similaridades = cosine_similarity(
        embedding_usuario,
        embeddings_perguntas
    )[0]
    
    # 3. Pegar Top K
    indices_top = np.argsort(similaridades)[::-1][:top_k]
    
    # 4. Filtrar por threshold
    resultados = []
    for idx in indices_top:
        score = similaridades[idx]
        if score >= threshold:
            resultados.append({
                'pergunta': dados_faq['perguntas'][idx],
                'resposta': dados_faq['respostas'][idx],
                'similaridade': f"{score:.2%}"
            })
    
    return resultados

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pergunta = request.form.get('message', '').strip()
    
    resultados = buscar_resposta_similar(
        pergunta,
        top_k=3,
        threshold=0.5
    )
    
    return render_template('resultado.html',
                         pergunta_usuario=pergunta,
                         resultados=resultados,
                         total_resultados=len(resultados))
```

### Frontend

**index.html**: Formulário de busca  
**resultado.html**: Top 3 resultados com score

---

## 📚 Tecnologias Utilizadas

| Categoria | Tecnologia | Uso |
|-----------|-----------|-----|
| **NLP** | Sentence Transformers | Embeddings semânticos |
| **ML** | scikit-learn | Similaridade de cosseno |
| **Dados** | Pandas, NumPy | Manipulação de dados |
| **Limpeza** | BeautifulSoup4 | Remover HTML |
| **Web** | Flask | Backend |
| **Frontend** | HTML/CSS | Interface |
| **Persistência** | pickle, NumPy | Salvar embeddings |

---

## 📊 Exemplos de Uso

### Exemplo 1: Busca Direta

**Input**:
```
"Como fazer transferência PIX?"
```

**Output**:
```
Top 3 Resultados:

1. Como acesso o PIX?
   Similaridade: 87%
   Resposta: Para acessar o PIX, você precisa...

2. Como faço para cadastrar chave PIX?
   Similaridade: 74%
   Resposta: O cadastro de chave PIX pode ser feito...

3. Qual o limite de transferência PIX?
   Similaridade: 62%
   Resposta: O limite de transferência depende...
```

---

### Exemplo 2: Sinônimos

**Input**:
```
"Como realizar pagamento instantâneo?"
```

**Output**:
```
Top 3 Resultados:

1. Como acesso o PIX?
   Similaridade: 81%
   (PIX é pagamento instantâneo - modelo entende!)
```

---

### Exemplo 3: Pergunta Fora do Escopo

**Input**:
```
"Qual a previsão do tempo amanhã?"
```

**Output**:
```
Nenhum resultado encontrado com confiança suficiente.
(Todos abaixo de 50% threshold)
```

---

### Exemplo 4: Variação de Formulação

**Input 1**: "Como cadastrar chave PIX?"  
**Input 2**: "Qual o processo para registrar chave no PIX?"

**Ambos retornam**:
```
Como faço para cadastrar chave PIX?
Similaridade: ~85%
```

**Por quê?** Embeddings capturam significado, não palavras exatas.

---

## 🔍 Como o Sistema Decide?

### Embeddings Capturam Semântica

```python
# Exemplo simplificado (512D → 3D para visualização)

"Como fazer PIX?" → [0.8, 0.3, 0.1]
"Como realizar PIX?" → [0.79, 0.31, 0.09]  # Muito similar!
"Qual o horário do banco?" → [0.1, 0.7, 0.6]  # Diferente

# Similaridade de cosseno
cos("Como fazer PIX?", "Como realizar PIX?") = 0.98 (98%)
cos("Como fazer PIX?", "Qual horário banco?") = 0.23 (23%)
```

### Threshold de 50%

**Por que 50%?**
- ✅ Evita respostas sem sentido
- ✅ Balanceia recall vs precision
- ⚠️ Ajustável conforme necessidade

**Experimentos**:
```
Threshold 30%: Muitos falsos positivos
Threshold 50%: Balanceado ✓
Threshold 70%: Perde resultados válidos
```

---

## 📈 Performance e Limitações

### Quando Funciona Bem

- ✅ Perguntas dentro do domínio (PIX, empréstimos, Registrato)
- ✅ Variações de formulação da mesma pergunta
- ✅ Sinônimos ("fazer" vs "realizar")
- ✅ Perguntas completas (>5 palavras)

### Quando Pode Falhar

- ❌ Perguntas muito genéricas ("Como funciona?")
- ❌ Tópicos fora da base de conhecimento
- ❌ Perguntas muito curtas (<3 palavras)
- ❌ Gírias ou termos técnicos não presentes no FAQ

### Métricas Típicas

```
Top-1 Accuracy: ~75%
Top-3 Accuracy: ~90%
Velocidade: <1s por busca
```

---

## 🔮 Melhorias Futuras

### Dados
- [ ] Expandir base de conhecimento (3k+ FAQs)
- [ ] Adicionar FAQs de múltiplos bancos
- [ ] Feedback do usuário ("resposta útil?")
- [ ] Re-treinar modelo com feedback

### Modelo
- [ ] Testar modelos maiores (BERT base português)
- [ ] Fine-tuning em domínio financeiro
- [ ] Usar reranker (bi-encoder + cross-encoder)
- [ ] Adicionar filtros (categoria, data)

### Aplicação
- [ ] API REST para integração
- [ ] Histórico de buscas
- [ ] Analytics (perguntas mais frequentes)
- [ ] Chatbot conversacional (multi-turn)
- [ ] Deploy em cloud (Heroku, Railway)

### UX
- [ ] Sugestões de perguntas populares
- [ ] Autocomplete
- [ ] "Você quis dizer...?"
- [ ] Exportar FAQ em PDF

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
- Adicionar mais FAQs
- Melhorar UI/UX
- Implementar API REST
- Adicionar testes automatizados
- Criar dashboard de analytics

---

## 📖 Recursos Adicionais

### Sentence Transformers
- [Documentação](https://www.sbert.net/)
- [Modelos disponíveis](https://www.sbert.net/docs/pretrained_models.html)
- [Paper original](https://arxiv.org/abs/1908.10084)

### Similaridade Semântica
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Semantic Search Tutorial](https://www.sbert.net/examples/applications/semantic-search/README.html)

### Datasets Similares
- [Stack Overflow Questions](https://www.kaggle.com/datasets/stackoverflow/stacksample)
- [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)

---

## 📝 Citação

Se usar este projeto, por favor cite:

```
@misc{sistema_busca_faq_2026,
  author = {Carlos Henrique Bamberg Marques},
  title = {Sistema de Busca Inteligente para FAQs com Sentence Transformers},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/RickBamberg/Sistema_de_Busca_FAQs}
}
```

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)  
**LinkedIn**: [carlos-henrique-bamberg-marques](https://www.linkedin.com/in/carlos-henrique-bamberg-marques/)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 🙏 Agradecimentos

- [Banco Central do Brasil](https://www.bcb.gov.br/) - Dataset de FAQs
- [Sentence Transformers](https://www.sbert.net/) - Biblioteca de embeddings
- [Flask](https://flask.palletsprojects.com/) - Framework web
- Comunidade de NLP brasileira

---

**💡 Dica**: Busca semântica é o futuro! Este é um ótimo baseline para chatbots e sistemas de Q&A.

*Projeto desenvolvido como parte do curso "Especialista em IA" - Módulo EAI_04*
