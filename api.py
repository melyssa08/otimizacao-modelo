from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Carregar o modelo salvo sem compilar
model = load_model('model_optimizer.h5', compile=False)

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Definir o tamanho da imagem de entrada esperado pelo modelo
input_shape = (32, 32)

# Criar uma instância do Flask
app = Flask(__name__)

# Definir a rota para a API
@app.route('/predict', methods=['POST'])
def predict():
    # Verificar se a requisição é do tipo POST
    if request.method == 'POST':
        # Verificar se foi enviado um arquivo na requisição
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        # Ler o arquivo enviado na requisição
        file = request.files['file']
        
        # Verificar se o arquivo tem extensão de imagem
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            # Carregar a imagem em formato PIL
            img = Image.open(io.BytesIO(file.read()))
            
            # Redimensionar a imagem para o tamanho esperado pelo modelo
            img = img.resize(input_shape)
            
            # Converter a imagem para um array numpy
            img_array = np.array(img) / 255.0
            
            # Adicionar uma dimensão para representar o batch de imagens
            img_array = np.expand_dims(img_array, axis=0)
            
            # Fazer a predição com o modelo
            prediction = model.predict(img_array)
            
            # Obter a categoria com maior probabilidade
            predicted_class = np.argmax(prediction)
            
            # Retornar a categoria como JSON
            return jsonify({'predicted_class': int(predicted_class)})

# Executar a aplicação Flask
if __name__ == '__main__':
    app.run()