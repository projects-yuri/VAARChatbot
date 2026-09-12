# VAARChatbot

Chatbot com arquitetura **RAG (Retrieval-Augmented Generation)** desenvolvido para responder perguntas com base na documentação do **Projeto PI-5**, com foco em **Fundeb, VAAR e ICMS Educacional do Rio Grande do Sul**.

## Como executar o projeto

### 1. Requisitos

Para executar o projeto localmente, é necessário ter:

* Python 3.11 ou superior
* Git
* Conexão com a internet
* Chave de API da OpenRouter

### 2. Clonar o repositório

Abra o **CMD**, **PowerShell** ou terminal do **VS Code** e execute:

```bash
git clone https://github.com/projects-yuri/VAARChatbot.git
```

Depois, entre na pasta do projeto:

```bash
cd VAARChatbot
```

### 3. Criar o ambiente virtual

No Windows:

```bash
python -m venv .venv
```

Depois, ative o ambiente:

**CMD:**

```bash
.venv\Scripts\activate
```

**PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

Se funcionar corretamente, o terminal deverá mostrar algo semelhante a:

```text
(.venv) C:\Users\usuario\VAARChatbot>
```

### 4. Instalar as dependências

Com o ambiente virtual ativado, execute:

```bash
pip install -r requirements.txt
```

As bibliotecas necessárias para o projeto estão listadas no arquivo `requirements.txt`.

### 5. Configurar a API da OpenRouter

Na raiz do projeto, crie um arquivo chamado `.env` e adicione:

```env
OPENROUTER_API_KEY=SUA_CHAVE_OPENROUTER
OPENROUTER_MODEL=z-ai/glm-5.2:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
TOP_K=5
```

Substitua `SUA_CHAVE_OPENROUTER` pela sua chave pessoal da OpenRouter.

> **Importante:** o arquivo `.env` não deve ser enviado ao GitHub, pois contém informações privadas de acesso à API. Ele deve estar incluído no `.gitignore`.

## Documentos utilizados

Os documentos utilizados pelo RAG estão armazenados na pasta:

```text
raw/
```

O sistema utiliza arquivos **PDF** e **TXT** como base de conhecimento.

Durante a indexação, os documentos são:

1. Lidos e convertidos em texto;
2. Divididos em trechos, chamados de **chunks**;
3. Transformados em **embeddings**;
4. Armazenados no banco vetorial **ChromaDB**.

## Criar o índice dos documentos

Antes de executar o chatbot pela primeira vez, execute:

```bash
python -m src.ingestion --reset
```

Esse comando:

* lê os documentos da pasta `raw/`;
* cria os chunks;
* gera os embeddings;
* cria ou recria o banco vetorial;
* salva o índice no diretório `data/chroma/`.

Esse comando também deve ser executado novamente caso os documentos da pasta `raw/` sejam alterados.

## Executar a interface web

Depois da indexação, execute:

```bash
streamlit run app.py
```

O Streamlit iniciará a aplicação localmente.

Normalmente, ela estará disponível em:

```text
http://localhost:8501
```

Abra esse endereço no navegador para utilizar o chatbot.

## Executar pelo terminal

Também é possível executar a versão em linha de comando:

```bash
python chatbot.py
```

## Funcionamento do RAG

O fluxo da aplicação funciona da seguinte forma:

```text
Pergunta do usuário
        ↓
Conversão da pergunta em embedding
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
```

Por padrão, o sistema utiliza:

```text
TOP_K=5
```

Isso significa que até **cinco trechos relevantes** podem ser recuperados para fornecer contexto ao modelo antes da geração da resposta.

## Estrutura principal do projeto

```text
VAARChatbot/
│
├── raw/
│
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
```

## Principais arquivos

* **`src/ingestion.py`** — leitura dos documentos, criação dos chunks, geração dos embeddings e armazenamento no ChromaDB.
* **`src/rag.py`** — recuperação dos trechos relevantes, criação do contexto e comunicação com o modelo de linguagem.
* **`src/settings.py`** — configurações gerais e variáveis de ambiente do projeto.
* **`app.py`** — interface web desenvolvida com Streamlit.
* **`chatbot.py`** — versão do chatbot executada pelo terminal.
* **`raw/`** — documentos utilizados como base de conhecimento.

## Tecnologias utilizadas

* **Python**
* **Streamlit**
* **LangChain**
* **OpenRouter**
* **Sentence Transformers**
* **ChromaDB**
* **PyPDF**

## Execução rápida

Após configurar o projeto, os principais comandos são:

```bash
git clone https://github.com/projects-yuri/VAARChatbot.git
cd VAARChatbot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.ingestion --reset
streamlit run app.py
```

Depois, acesse:

```text
http://localhost:8501
```

## Observação

Na primeira execução, a geração dos embeddings pode levar mais tempo devido ao carregamento do modelo e ao processamento dos documentos.

Depois que o índice vetorial for criado, o chatbot poderá reutilizar o banco existente para realizar novas consultas.
