from datasets import load_dataset
import random
import json
from langchain_community.llms import Ollama
import time

def translate_text(text, client):
    prompt = f"Traduza o seguinte texto do inglês para o português, não traduza código:\n\n{text}\n\nTradução:"
    response = client.generate([prompt], model_kwargs={"max_tokens": 500})
    return response.generations[0][0].text.strip()

def generate_datasets():
    # Inicializar o cliente Ollama
    client = Ollama(model="llama3.1PT")

    # Carregar o dataset MetaMathQA
    dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")

    # Acessar o split de treinamento
    train_data = dataset['train']

    # Selecionar 100 amostras aleatórias
    samples = random.sample(range(len(train_data)), 100)

    # Criar listas para armazenar os dados de treinamento e teste
    train_set = []
    test_set = []

    # Dividir as amostras em conjuntos de treinamento (80) e teste (20)
    for i, idx in enumerate(samples):
        item = train_data[idx]
        
        # Verificar se as chaves 'instruction' e 'output' existem
        if 'instruction' in item and 'output' in item:
            translated_question = translate_text(item['instruction'], client)
            translated_answer = translate_text(item['output'], client)
            
            data_point = {
                "question_en": item['instruction'],
                "question_pt": translated_question,
                "answer_en": item['output'],
                "answer_pt": translated_answer
            }
            
            if i < 80:
                train_set.append(data_point)
            else:
                test_set.append(data_point)
            
            # Adicionar um pequeno atraso para evitar sobrecarregar o servidor Ollama
            time.sleep(0.2)
            
            # Imprimir progresso
            print(f"Processado {i+1}/100 itens")
        else:
            print(f"Item {i} não contém 'instruction' ou 'output'. Pulando.")

    # Salvar os conjuntos de dados em arquivos JSON
    with open('code_train_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)

    with open('code_test_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(f"Conjunto de treinamento gerado com {len(train_set)} amostras.")
    print(f"Conjunto de teste gerado com {len(test_set)} amostras.")

if __name__ == "__main__":
    generate_datasets()