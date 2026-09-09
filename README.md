# VAARChatbot

Chatbot com arquitetura RAG (Retrieval-Augmented Generation) desenvolvido para responder perguntas com base na documentação do Projeto PI-5, com foco em Fundeb, VAAR e ICMS Educacional do Rio Grande do Sul.

## Como executar o projeto

### 1. Requisitos

Para executar o projeto localmente, é necessário ter:

- Python 3.11 ou superior
- Git
- Conexão com a internet
- Chave de API da OpenRouter

### 2. Clonar o repositório

Abra o CMD, PowerShell ou terminal do VS Code e execute:

git clone https://github.com/projects-yuri/VAARChatbot.git

Depois entre na pasta:

cd VAARChatbot

### 3. Criar ambiente virtual

No Windows:

python -m venv .venv

Depois ative:

.venv\Scripts\activate

Se funcionar corretamente, o terminal deverá mostrar algo semelhante a:

(.venv) C:\Users\usuario\VAARChatbot>

### 4. Instalar as dependências

Com o ambiente virtual ativado, execute:

pip install -r requirements.txt

As bibliotecas utilizadas pelo projeto estão listadas no arquivo requirements.txt.

### 5. Configurar a API da OpenRouter

Na raiz do projeto, crie um arquivo chamado:

.env

Adicione:

OPENROUTER_API_KEY=SUA_CHAVE_OPENROUTER
OPENROUTER_MODEL=z-ai/glm-5.2:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
TOP_K=5

Substitua SUA_CHAVE_OPENROUTER pela sua chave pessoal.

O arquivo .env não deve ser enviado ao GitHub, pois contém informações privadas de acesso à API.

### 6. Documentos utilizados

Os documentos utilizados pelo RAG estão armazenados na pasta:

raw/

O sistema utiliza arquivos PDF e TXT como base de conhecimento.

Durante a indexação, os documentos são:

- lidos;
- convertidos em texto;
- divididos em trechos;
- transformados em embeddings;
- armazenados no banco vetorial ChromaDB.

### 7. Criar o índice dos documentos

Antes de executar o chatbot pela primeira vez, execute:

python -m src.ingestion --reset

Esse comando recria o índice vetorial a partir dos documentos existentes na pasta raw/.

O índice é armazenado localmente no diretório:

data/chroma/

### 8. Executar a interface web

Depois da indexação, execute:

streamlit run app.py

O Streamlit iniciará o chatbot localmente.

Normalmente, a aplicação estará disponível no endereço:

http://localhost:8501

Abra esse endereço no navegador.

### 9. Executar pelo terminal

Também é possível executar a versão em linha de comando:

python chatbot.py

## Funcionamento do RAG

O fluxo da aplicação funciona da seguinte forma:

Pergunta do usuário
↓
Busca vetorial no ChromaDB
↓
Recuperação dos trechos mais relevantes
↓
Envio dos trechos como contexto para o modelo de linguagem
↓
Geração da resposta
↓
Exibição da resposta e das fontes utilizadas

Por padrão, o sistema utiliza TOP_K=5, recuperando até cinco trechos relevantes para cada pergunta.

## Estrutura principal do projeto

VAARChatbot/
│
├── raw/
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── rag.py
│   └── settings.py
│
├── app.py
├── chatbot.py
├── requirements.txt
├── .gitignore
└── README.md

## Principais arquivos

src/ingestion.py
Responsável pela leitura dos documentos, criação dos chunks, geração dos embeddings e armazenamento no ChromaDB.

src/rag.py
Responsável pela recuperação dos trechos relevantes e envio do contexto para o modelo de linguagem.

src/settings.py
Responsável pelas configurações do projeto.

app.py
Interface web desenvolvida com Streamlit.

chatbot.py
Versão do chatbot executada diretamente pelo terminal.

raw/
Pasta que contém os documentos utilizados como base de conhecimento.

## Execução rápida

Depois que o projeto estiver configurado, os principais comandos são:

git clone https://github.com/projects-yuri/VAARChatbot.git
cd VAARChatbot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.ingestion --reset
streamlit run app.py

Depois acesse:

http://localhost:8501
