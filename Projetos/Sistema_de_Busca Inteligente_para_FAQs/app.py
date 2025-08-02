from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- INICIALIZAÇÃO E CARREGAMENTO DOS MODELOS ---

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = 'sua_chave_secreta_pode_ser_qualquer_coisa'

# Caminhos para os arquivos
MODELS_DIR = 'models'
DADOS_PATH = os.path.join(MODELS_DIR, 'dados_faq.pkl')
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, 'embeddings_faq.npy')

# Carregamento dos dados e modelo
try:
    with open(DADOS_PATH, 'rb') as f:
        dados_faq = pickle.load(f)
    embeddings_perguntas = np.load(EMBEDDINGS_PATH)
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    print("✅ Dados e modelo carregados com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar os dados: {e}")
    dados_faq = None
    embeddings_perguntas = None
    model = None

def buscar_resposta_similar(pergunta_usuario, top_k=3, threshold=0.5): # Adicionado threshold
    """
    Busca as respostas mais similares que atinjam um limiar de confiança.
    """
    if not dados_faq or embeddings_perguntas is None or not model:
        return []
    
    try:
        embedding_usuario = model.encode([pergunta_usuario])
        similaridades = cosine_similarity(embedding_usuario, embeddings_perguntas)[0]
        
        # Encontra os índices das top_k perguntas mais similares
        indices_similares_brutos = np.argsort(similaridades)[::-1][:top_k]
        
        # *** A MUDANÇA MAIS IMPORTANTE ESTÁ AQUI ***
        # Filtra os resultados, mantendo apenas aqueles com similaridade >= threshold
        resultados = []
        for i, idx in enumerate(indices_similares_brutos):
            score = similaridades[idx]
            if score >= threshold:
                resultado = {
                    'posicao': len(resultados) + 1,
                    'pergunta': dados_faq['perguntas'][idx],
                    'resposta': dados_faq['respostas'][idx],
                    'similaridade': f"{score:.2%}"
                }
                resultados.append(resultado)
        
        return resultados
        
    except Exception as e:
        print(f"Erro na busca: {e}")
        return []

# --- ROTAS DA APLICAÇÃO ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not dados_faq or embeddings_perguntas is None or not model:
        return "Erro: O sistema de FAQ não está disponível.", 500

    pergunta = request.form.get('message', '').strip()
    
    if not pergunta:
        return render_template('index.html', error="Por favor, digite sua dúvida para buscar no FAQ.")

    try:
        # Busca as respostas mais similares usando o threshold
        # Você pode ajustar o valor do threshold aqui se quiser
        resultados = buscar_resposta_similar(pergunta, top_k=3, threshold=0.5)
        
        # O template 'resultado.html' já lida com o caso de 'resultados' ser uma lista vazia
        return render_template(
            'resultado.html', 
            pergunta_usuario=pergunta,
            resultados=resultados,
            total_resultados=len(resultados)
        )
    
    except Exception as e:
        print(f"Erro durante a busca: {e}")
        return render_template('index.html', error="Ocorreu um erro interno durante a busca. Tente novamente.")

if __name__ == '__main__':     
    app.run(debug=True)