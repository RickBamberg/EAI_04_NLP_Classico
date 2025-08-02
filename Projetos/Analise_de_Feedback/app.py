from flask import Flask, render_template, request
from joblib import load
import os

# --- INICIALIZAÇÃO E CARREGAMENTO DOS MODELOS ---

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = 'sua_chave_secreta_pode_ser_qualquer_coisa'

# Caminhos para os dois modelos
MODELS_DIR = 'models'
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_pipeline.pkl')
SUGGESTION_MODEL_PATH = os.path.join(MODELS_DIR, 'suggestion_pipeline.pkl')

# Carregamento único dos dois modelos
try:
    pipeline_sentimento = load(SENTIMENT_MODEL_PATH)
    pipeline_sugestao = load(SUGGESTION_MODEL_PATH)
    print("✅ Ambos os modelos foram carregados com sucesso!")
except Exception as e:
    print(f"❌ Erro fatal ao carregar os modelos: {e}")
    pipeline_sentimento = None
    pipeline_sugestao = None

# --- ROTAS DA APLICAÇÃO ---

@app.route('/')
def home():
    """Página inicial com o formulário."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Processa a predição para ambos os modelos e retorna os resultados."""
    # Verifica se os modelos foram carregados corretamente
    if not pipeline_sentimento or not pipeline_sugestao:
        return "Erro: Os modelos de análise não estão disponíveis.", 500

    # 1. Recebe o texto que o usuário enviou via formulário
    message = request.form.get('message', '').strip()
    
    # Se o texto estiver vazio, retorna para a página inicial com erro
    if not message:
        return render_template('index.html', error="Por favor, digite um texto para analisar.")

    try:
        # --- ANÁLISE DE SENTIMENTO (MODELO 1) ---
        sentiment_code = pipeline_sentimento.predict([message])[0]
        sentiment_map = {0: 'Negativo', 1: 'Positivo'}
        sentiment = sentiment_map.get(sentiment_code, 'Desconhecido')
        
        # Obtém a confiança do modelo de sentimento
        sentiment_probabilities = pipeline_sentimento.predict_proba([message])[0]
        sentiment_confidence = round(max(sentiment_probabilities) * 100, 2)

        # --- DETECÇÃO DE SUGESTÃO (MODELO 2) ---
        suggestion_code = pipeline_sugestao.predict([message])[0]
        
        # Mapeia o resultado para um texto amigável
        is_suggestion = "Sim" if suggestion_code == 1 else "Não"
        
        # Obtém a confiança do modelo de sugestão
        suggestion_probabilities = pipeline_sugestao.predict_proba([message])[0]
        suggestion_confidence = round(max(suggestion_probabilities) * 100, 2)

        # 5. Retorna a página de resultado com TODAS as informações
        return render_template(
            'resultado.html', 
            review=message,
            sentiment_prediction=sentiment,
            sentiment_confidence=sentiment_confidence,
            is_suggestion=is_suggestion,
            suggestion_confidence=suggestion_confidence
        )
    
    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return render_template('index.html', error="Ocorreu um erro interno durante a análise. Por favor, tente novamente.")

if __name__ == '__main__':     
    app.run(debug=True)
