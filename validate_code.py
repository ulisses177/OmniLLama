import json
import time
import os
import sys
import re
from io import StringIO
from contextlib import redirect_stdout
from core import client, formatar_passos

# Adiciona o diretório atual ao sys.path para importar app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Carregar os datasets
with open('code_train_dataset.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('code_test_dataset.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Processar o conjunto de teste
for item in test_data:
    print(f"Processando: {item['question_pt']}")
    
    # Gerar a cadeia de raciocínio
    steps, total_time = gerar_cadeia_raciocinio(item['question_pt'])
    
    # Obter a resposta correta do treinamento
    resposta_correta = item['answer_pt']
    
    # Gerar a resposta com base na cadeia de raciocínio
    resposta_gerada = resposta_final_llm(item['question_pt'], steps)
    
    # Validar a cadeia de raciocínio
    if validar_cadeia_raciocinio(item['question_pt'], resposta_gerada, resposta_correta):
        print("Cadeia de raciocínio validada.")
        
        # Adicionar a resposta e passos ao conjunto de treinamento
        train_data.append({
            'question_pt': item['question_pt'],
            'answer_pt': resposta_correta,
            'steps': steps
        })
    else:
        print("Cadeia de raciocínio não validada.")
    
    time.sleep(0.1)  # Para evitar sobrecarregar o servidor Ollama

# Salvar o conjunto de treinamento atualizado
with open('code_train_dataset_updated.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

print("Processamento concluído.")