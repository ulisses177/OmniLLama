import os
import faiss
import re
import signal
from io import StringIO
import asyncio
from contextlib import redirect_stdout
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.llms import Ollama
from langchain_community.docstore.in_memory import InMemoryDocstore
import uuid

from agent import Agent  # Importa a classe Agent
from blocks.base import ReasoningBlock
from blocks.structure import StructureDataBlock
from blocks.analyze import AnalyzeDataBlock
from blocks.code_generation import CodeGenerationBlock
from blocks.synthesis import SynthesisBlock  # Novo bloco para síntese

VECTORSTORE_DIR = "vectorstore"

# Custom timeout exception
class TimeoutException(Exception):
    pass

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutException()

def load_or_create_vectorstore():
    if os.path.exists(VECTORSTORE_DIR):
        try:
            vectorstore = FAISS.load_local(VECTORSTORE_DIR, HuggingFaceEmbeddings())
            print("Vectorstore carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar o vectorstore existente: {e}")
            print("Criando um novo vectorstore.")
            vectorstore = create_vectorstore()
    else:
        print("Vectorstore não encontrado. Criando um novo vectorstore.")
        vectorstore = create_vectorstore()
    return vectorstore

def create_vectorstore():
    embedding_model = HuggingFaceEmbeddings()
    embedding_dimension = len(embedding_model.embed_query("teste"))
    index = faiss.IndexFlatL2(embedding_dimension)
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}
    vectorstore = FAISS(embedding_function=embedding_model, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    vectorstore.save_local(VECTORSTORE_DIR)
    print("Novo vectorstore criado e salvo.")
    return vectorstore

# Passo 1: Inicializar o LLM Ollama
llm = Ollama(model="llama3.1:8b-instruct-fp16")

# Passo 2: Criar ou carregar o FAISS vectorstore para armazenar contexto ou trechos de código
vectorstore = load_or_create_vectorstore()

# Passo 3: Definir o prompt e a cadeia do agente
system_message = SystemMessagePromptTemplate.from_template(
    "Você é uma assistente de IA que gera código Python. "
    "Por favor, forneça apenas o bloco de código entre três crases. "
    "Você não deve incluir bibliotecas externas como numpy, pandas, etc. "
    "Sempre tente garantir que os programas funcionem corretamente e em no máximo 30 segundos eles devem ser terminados. "
    "Qualquer descrição ou informação adicional deve ser retornada como metadados."
)
human_message = HumanMessagePromptTemplate.from_template("Query: {combined_input}")

prompt = ChatPromptTemplate.from_messages([system_message, human_message])

def combine_input(context, query, short_term_memory):
    return f"Context: {context}\nShort-Term Memory: {short_term_memory}\nQuestion: {query}"

# Short-Term Memory para manter os últimos 5 comandos
short_term_memory = []

# Passo 4: Funções auxiliares para parse, salvar, executar e armazenar código com tratamento de erros
def parse_code_from_response(response):
    code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
    return code_blocks

def execute_code(code):
    with StringIO() as buf, redirect_stdout(buf):
        try:
            exec(code)
            output = buf.getvalue()
        except Exception as e:
            output = str(e)
            code += f"\n\n# Error: {output}"  # Adiciona o erro ao código
    return output, code

def save_code_to_cache(code):
    file_id = str(uuid.uuid4())
    file_name = f"{file_id}.py"
    file_path = os.path.join("cache", file_name)
    
    os.makedirs("cache", exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(code)
    
    return file_path

def chunk_and_store_code(code, vectorstore, description):
    chunk_size = 512
    for i in range(0, len(code), chunk_size):
        chunk = code[i:i + chunk_size]
        doc_id = str(uuid.uuid4())
        metadata = {"description": description}
        vectorstore.add_texts([chunk], ids=[doc_id], metadatas=[metadata])

def save_vectorstore(vectorstore, directory=VECTORSTORE_DIR):
    os.makedirs(directory, exist_ok=True)
    vectorstore.save_local(directory)

# Passo 5: Definir uma função assíncrona para interagir com o agente
async def run_agent(query: str, agent: Agent, vectorstore: FAISS):
    context_documents = vectorstore.similarity_search(query)
    context_chunks = [doc.page_content for doc in context_documents]
    context_metadata = [doc.metadata for doc in context_documents]
    context = "\n".join(context_chunks)

    combined_input = combine_input(context, query, "\n".join(short_term_memory))
    
    results = await agent.process(combined_input)
    
    synthesized_response = results.get('SynthesisBlock', {}).get('synthesized_response', 'Resposta não disponível.')
    print(f"Resposta Sintetizada: {synthesized_response}")
    
    short_term_memory.append(query)
    if len(short_term_memory) > 5:
        short_term_memory.pop(0)
    
    save_vectorstore(vectorstore)
    
    return synthesized_response

# Passo 6: Ponto de entrada principal para executar o agente
if __name__ == "__main__":
    agent = Agent()  # Inicializa o agente
    while True:
        user_input = input("Enter your request: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        synthesized_response = asyncio.run(run_agent(user_input, agent, vectorstore))
        print(f"Resposta Final: {synthesized_response}")
