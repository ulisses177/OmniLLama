import os
import sys

# Adiciona o diretório atual ao sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import gradio as gr
import json
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import re
import torch
import faiss
from models.diffusion_embedding import DiffusionEmbedding
from train_diffusion import treinar_diffusion_model
from sentence_transformers import SentenceTransformer
from validate_code import validar_cadeia_raciocinio
from core import client, formatar_passos, resposta_final_llm, obter_resposta_correta

# Inicialize o modelo de embedding fora da função para reutilização
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def extrair_dados_do_texto(texto):
    padrao = r"Passo (\d+): (.*?)\nConteúdo: ([\s\S]*?)(?:\nPróxima ação: (.*?))?(?:\n|$)"
    matches = re.finditer(padrao, texto, re.DOTALL)
    passos = []
    for match in matches:
        numero = match.group(1)
        titulo = match.group(2)
        conteudo = match.group(3) or ""
        proxima_acao = match.group(4) or "continuar"
        passos.append({
            "title": f"Passo {numero}: {titulo}",
            "content": conteudo.strip(),
            "next_action": proxima_acao.strip().lower()
        })
    return passos

def salvar_embeddings_faiss(embeddings, index_path, embedding_dim):
    model_path = 'diffusion_model.pth'
    
    # Verificar se o modelo já existe
    if os.path.exists(model_path):
        print("Carregando modelo existente...")
        model = DiffusionEmbedding(embedding_dim)
        model.load_state_dict(torch.load(model_path))
    else:
        print("Criando e treinando novo modelo...")
        model = treinar_diffusion_model(embeddings, embedding_dim)
        torch.save(model.state_dict(), model_path)
    
    # Codificar os embeddings com o modelo
    model.eval()
    tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        encoded_embeddings = model.encode(tensor_embeddings).cpu().numpy()
    
    # Criar e salvar o índice FAISS
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(encoded_embeddings)
    
    # Garantir que o diretório de destino exista
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    print(f"Tentando salvar o índice FAISS em: {index_path}")
    try:
        faiss.write_index(index, index_path)
        print(f"Índice FAISS salvo com sucesso em {index_path}")
    except Exception as e:
        print(f"Erro ao salvar o índice FAISS: {e}")
        print(f"Conteúdo do diretório {os.path.dirname(index_path)}:")
        print(os.listdir(os.path.dirname(index_path)))
    
    print(f"Modelo e índice FAISS processados.")

def recuperar_embeddings_faiss(query, index_path, embedding_dim):
    model_path = 'diffusion_model.pth'
    
    # Verificar se o modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modelo de difusão não encontrado. Execute salvar_embeddings_faiss primeiro.")
    
    # Carregar o modelo
    model = DiffusionEmbedding(embedding_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Carregar o índice FAISS
    index = faiss.read_index(index_path)
    
    # Codificar a query
    query_tensor = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        encoded_query = model.encode(query_tensor).cpu().numpy()
    
    # Buscar no índice FAISS
    _, indices = index.search(encoded_query, k=1)
    retrieved_embedding = index.reconstruct(indices[0][0])
    
    # Decodificar o embedding recuperado
    retrieved_tensor = torch.tensor(retrieved_embedding, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        decoded_embedding = model.decode(retrieved_tensor).cpu().numpy()
    
    return decoded_embedding

def gerar_cadeia_raciocinio(prompt):
    # Prompt Inicial
    prompt_inicial = f"""Você é um assistente AI especializado em programação Python. Seu objetivo é resolver o seguinte problema:
    
    {prompt}
    
    Forneça uma cadeia de raciocínio estruturada com os blocos a seguir:
    
    1. **Prompt Inicial**: Definição do problema e objetivos.
    2. **Etapas Intermediárias**: Passos detalhados para resolver o problema.
    3. **Verificação**: Validação da lógica e execução do código gerado.
    4. **Resposta Final**: Resumo e conclusão.
    
    Siga o formato abaixo para cada bloco:
    
    ---
    **Bloco X: Nome do Bloco**
    Conteúdo detalhado do bloco.
    ---
    
    Exemplo de formatação:
    
    ---
    **Bloco 1: Prompt Inicial**
    Definição clara do problema a ser resolvido.
    ---
    """
    
    # Inicialização das variáveis
    steps = []
    step_count = 1
    total_thinking_time = 0

    try:
        # Geração da cadeia de raciocínio inicial
        print("Início da geração da cadeia de raciocínio para o prompt inicial:", prompt)
        start_time = time.time()
        response = client.generate([prompt_inicial], model_kwargs={"max_tokens": 1000})
        
        if not response.generations or not response.generations[0] or not response.generations[0][0].text.strip():
            print("Erro: resposta vazia ou inválida da API.")
            return steps, total_thinking_time
        
        resposta = response.generations[0][0].text.strip()
        print("Resposta inicial recebida:", resposta)
        
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append(("Bloco 1: Prompt Inicial", resposta, thinking_time))
        
        # Extrair e adicionar etapas intermediárias
        etapas_intermediarias = extrair_blocos(resposta, "Etapas Intermediárias")
        for etapa in etapas_intermediarias:
            steps.append(etapa)
            step_count += 1
            if step_count > 10:  # Limite para evitar loops infinitos
                break

        # Verificação
        bloco_verificacao = gerar_verificacao(prompt, steps)
        steps.append(bloco_verificacao)

        # Resposta Final
        bloco_final = gerar_resposta_final(prompt, steps)
        steps.append(bloco_final)

    except Exception as e:
        print(f"Erro durante a geração da cadeia de raciocínio: {e}")
        return steps, total_thinking_time

    return steps, total_thinking_time

def extrair_blocos(resposta, nome_bloco):
    """
    Função para extrair blocos específicos da resposta.
    """
    padrao = rf"---\n\*\*{nome_bloco}\*\*\n([\s\S]+?)\n---"
    matches = re.findall(padrao, resposta)
    blocos = []
    for match in matches:
        blocos.append((nome_bloco, match.strip(), 0))
    return blocos

def gerar_verificacao(prompt, steps):
    """
    Função para gerar o bloco de verificação.
    """
    # Formatar os passos para validação
    cadeia_pensamentos = formatar_passos(steps)
    
    # Obter a resposta correta
    resposta_correta = obter_resposta_correta(prompt)
    
    if not resposta_correta:
        return ("Bloco 3: Verificação", "Falha ao obter a resposta correta para verificação.", 0)
    
    # Executar a validação usando a função de validate_code.py
    validacao = validar_cadeia_raciocinio(prompt, cadeia_pensamentos, resposta_correta)
    
    return ("Bloco 3: Verificação", validacao, 0)

def gerar_resposta_final(prompt, steps):
    """
    Função para gerar o bloco de resposta final.
    """
    mensagem = f"""Com base na seguinte cadeia de raciocínio e na verificação realizada, forneça uma **resposta final** concisa para o prompt original:
    
    Prompt original: {prompt}
    
    Cadeia de Raciocínio:
    {formatar_passos(steps)}
    
    Resposta Final:
    """
    resposta_final = client.generate([mensagem], model_kwargs={"max_tokens": 500}).generations[0][0].text.strip()
    return ("Bloco 4: Resposta Final", resposta_final, 0)

def converter_resposta_para_embedding(resposta):
    return sentence_model.encode([resposta])[0]

def aprovar_resposta(resposta):
    embedding = converter_resposta_para_embedding(resposta)
    index_dir = os.path.join(os.getcwd(), "faiss_index")
    index_filename = "embeddings.index"
    index_path = os.path.join(index_dir, index_filename)
    os.makedirs(index_dir, exist_ok=True)
    print(f"Diretório do índice FAISS: {index_dir}")
    print(f"Caminho completo do arquivo de índice: {index_path}")
    salvar_embeddings_faiss([embedding], index_path, embedding_dim=384)  # 384 é a dimensão padrão do modelo all-MiniLM-L6-v2
    return f"Resposta aprovada e salva no FAISS em {index_path}!"

def gradio_interface(prompt):
    steps, total_time = gerar_cadeia_raciocinio(prompt)
    
    if not steps:
        return "Nenhuma resposta gerada. Pode ter ocorrido um erro.", "", f"Tempo total: {total_time:.2f} segundos"
    
    cadeia_pensamentos = formatar_passos(steps)
    resposta_final = resposta_final_llm(prompt, steps)
    
    return resposta_final, cadeia_pensamentos, f"Tempo total: {total_time:.2f} segundos"

# Definindo a interface com Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Cadeia de Raciocínio com Ollama e FAISS")
    
    prompt_input = gr.Textbox(label="Digite sua pergunta", placeholder="Exemplo: Quantas letras 'R' tem em 'morango'?", lines=2)
    
    output_text = gr.Textbox(label="Resposta Final", lines=5)
    cadeia_pensamentos = gr.Textbox(label="Cadeia de Pensamentos", lines=20)  # Aumentar o número de linhas
    status_text = gr.Textbox(label="Status", lines=2)

    submit_button = gr.Button("Gerar Resposta")
    aprovar_button = gr.Button("Aprovar Resposta")
    
    submit_button.click(fn=gradio_interface, 
                        inputs=[prompt_input], 
                        outputs=[output_text, cadeia_pensamentos, status_text])
    
    aprovar_button.click(fn=aprovar_resposta,
                         inputs=[output_text],
                         outputs=[status_text])

# Executando a interface Gradio
demo.launch()