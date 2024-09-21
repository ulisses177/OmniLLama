# OmniLLama

## Visão Geral

Este projeto implementa uma **Cadeia de Raciocínio** utilizando **Ollama** e **FAISS** para resolver problemas de programação em Python de maneira eficiente e estruturada. A aplicação utiliza modelos de embeddings para melhorar a relevância das respostas e oferece uma interface interativa através do **Gradio**.

## Funcionalidades

- **Geração de Cadeia de Raciocínio**: Estrutura as respostas em blocos lógicos para facilitar a compreensão e validação.
- **Embeddings com Diffusion Models**: Utiliza modelos de difusão para codificar e decodificar embeddings, melhorando a qualidade das buscas no índice FAISS.
- **Indexação com FAISS**: Armazena e recupera embeddings de maneira eficiente para consultas rápidas.
- **Interface Interativa com Gradio**: Fornece uma interface amigável para interação com o modelo, permitindo a geração e aprovação de respostas.
- **Validação de Respostas**: Inclui mecanismos para validar a lógica e a execução do código gerado.

## Estrutura do Projeto

    ├── app.py
    ├── models
    │ └── diffusion_embedding.py
    ├── train_diffusion.py
    ├── validate_code.py
    ├── core.py
    ├── requirements.txt
    └── README.md

### Descrição dos Arquivos

- **app.py**: Arquivo principal que configura a interface Gradio, define as funções de geração de raciocínio e interage com os módulos de embeddings e FAISS.
  
- **models/diffusion_embedding.py**: Define a classe `DiffusionEmbedding`, um modelo de difusão para codificação e decodificação de embeddings.
  
- **train_diffusion.py**: Contém a função `treinar_diffusion_model` para treinar o modelo de difusão com os embeddings fornecidos.
  
- **validate_code.py**: Funções para validar a cadeia de raciocínio gerada.
  
- **core.py**: Contém funções auxiliares para formatação de passos, obtenção de respostas corretas e processamento final das respostas.
  
- **requirements.txt**: Lista de dependências necessárias para executar o projeto.
  
- **README.md**: Este arquivo.

## Instalação

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/seu-usuario/cadeia-de-raciocinio.git
   cd cadeia-de-raciocinio
   ```

2. **Crie um ambiente virtual (opcional, mas recomendado):**

   ```bash
   python -m venv env
   source env/bin/activate  # No Windows: env\Scripts\activate
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. **Treine o modelo de difusão (se necessário):**

   Caso você já possua embeddings pré-treinados, pode pular este passo. Caso contrário, treine o modelo de difusão com os embeddings desejados.

   ```bash
   python train_diffusion.py
   ```

2. **Execute a aplicação:**

   ```bash
   python app.py
   ```

3. **Acesse a interface Gradio:**

   Após executar o `app.py`, uma interface Gradio será lançada automaticamente no seu navegador padrão. Se não abrir automaticamente, verifique o endereço URL fornecido no terminal.

4. **Interaja com a aplicação:**

   - **Digite sua pergunta** na caixa de texto fornecida.
   - **Clique em "Gerar Resposta"** para obter a resposta final, a cadeia de pensamentos e o tempo total de processamento.
   - **Aprove a resposta** clicando em "Aprovar Resposta" para salvar o embedding no índice FAISS.

## Dependências

As principais bibliotecas utilizadas neste projeto incluem:

- [Gradio](https://gradio.app/)
- [Langchain Community](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyTorch](https://pytorch.org/)
- [SentenceTransformers](https://www.sbert.net/)
- [Torch](https://pytorch.org/)

Todas as dependências podem ser instaladas através do arquivo `requirements.txt`.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir **issues** ou enviar **pull requests** para melhorar este projeto.

## Licença

Este projeto está licenciado sob a licença [MIT](LICENSE).

## Contato

Para dúvidas ou sugestões, entre em contato através do [email](mailto:seu-email@exemplo.com).

