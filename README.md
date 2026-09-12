# VAARChatbot

Chatbot com arquitetura **RAG (Retrieval-Augmented Generation)** desenvolvido para responder perguntas com base na documentação do **Projeto PI-5**, com foco em **Fundeb, VAAR e ICMS Educacional do Rio Grande do Sul**.

O sistema recupera informações relevantes dos documentos do projeto por meio de busca vetorial e utiliza esses trechos como contexto para gerar respostas mais relacionadas à documentação utilizada.

---

## Como executar o projeto

### 1. Requisitos

Para executar o projeto localmente, é necessário ter:

* Python 3.11 ou superior
* Git
* Conexão com a internet
* Chave de API da OpenRouter

---

### 2. Clonar o repositório

Abra o **CMD**, **PowerShell** ou terminal do **Visual Studio Code** e execute:

```bash
git clone https://github.com/projects-yuri/VAARChatbot.git
```

Depois, entre na pasta do projeto:

```bash
cd VAARChatbot
```

---

### 3. Criar o ambiente virtual

Execute:

```bash
python -m venv .venv
```

Depois, ative o ambiente virtual.

#### CMD

```cmd
.venv\Scripts\activate
```

#### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativado corretamente, o terminal deverá mostrar algo semelhante a:

```text
(.venv) C:\Users\usuario\VAARChatbot>
```

---

### 4. Instalar as dependências

Com o ambiente virtual ativado, execute:

```bash
pip install -r requirements.txt
```

Todas as bibliotecas necessárias para executar o projeto estão disponíveis no arquivo `requirements.txt`.

---

### 5. Configurar a API da OpenRouter

Na raiz do projeto, crie um arquivo chamado:

```text
.env
```

Dentro do arquivo, adicione:

```env
OPENROUTER_API_KEY=SUA_CHAVE_OPENROUTER
OPENROUTER_MODEL=z-ai/glm-5.2:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
TOP_K=5
```

Substitua:

```text
SUA_CHAVE_OPENROUTER
```

pela sua própria chave da OpenRouter.

> **Importante:** o arquivo `.env` contém informações privadas e não deve ser enviado ao GitHub.

---

## Documentos utilizados

Os documentos utilizados pelo sistema RAG estão armazenados na pasta:

```text
raw/
```

O sistema utiliza arquivos **PDF** e **TXT** como base de conhecimento.

Durante a indexação, os documentos passam pelas seguintes etapas:

1. Leitura dos arquivos;
2. Extração do conteúdo em texto;
3. Divisão do texto em pequenos trechos, chamados de **chunks**;
4. Transformação dos chunks em **embeddings**;
5. Armazenamento dos vetores no **ChromaDB**.

Esses vetores permitem localizar os trechos mais relacionados à pergunta realizada pelo usuário.

---

## Criar o índice dos documentos

Antes da primeira execução do chatbot, execute:

```bash
python -m src.ingestion --reset
```

Esse comando:

* lê os documentos da pasta `raw/`;
* extrai os textos;
* cria os chunks;
* gera os embeddings;
* cria o banco vetorial;
* salva o índice localmente.

O índice é armazenado no diretório:

```text
data/chroma/
```

Caso os documentos da pasta `raw/` sejam alterados, a indexação pode ser executada novamente com o mesmo comando:

```bash
python -m src.ingestion --reset
```

---

## Executar o chatbot

Após instalar as dependências, configurar a API e criar o índice dos documentos, execute:

```bash
streamlit run app.py
```

O Streamlit iniciará a aplicação localmente.

Normalmente, o endereço será:

```text
http://localhost:8501
```

Abra esse endereço no navegador para utilizar o chatbot.

---

# Funcionamento do RAG

O fluxo principal da aplicação funciona da seguinte forma:

```text
Pergunta do usuário
        ↓
Conversão da pergunta em embedding
        ↓
Busca vetorial no ChromaDB
        ↓
Recuperação dos trechos mais relevantes
        ↓
Montagem do contexto
        ↓
Envio do contexto e da pergunta ao modelo de linguagem
        ↓
Geração da resposta
        ↓
Exibição da resposta e das fontes utilizadas
```

Por padrão, o sistema utiliza:

```text
TOP_K=5
```

Isso significa que o mecanismo de recuperação busca até **cinco trechos relevantes** para utilizar como contexto na geração da resposta.

---

# Estrutura principal do projeto

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
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Principais arquivos

### `src/ingestion.py`

Responsável pelo processo de preparação dos documentos para o RAG:

* leitura dos arquivos;
* extração do texto;
* criação dos chunks;
* geração dos embeddings;
* armazenamento dos vetores no ChromaDB.

### `src/rag.py`

Responsável pela lógica principal do sistema RAG:

* recebe a pergunta do usuário;
* realiza a busca vetorial;
* recupera os trechos mais relevantes;
* cria o contexto;
* envia a pergunta e o contexto ao modelo de linguagem;
* retorna a resposta gerada.

### `src/settings.py`

Responsável pelas configurações gerais do projeto, incluindo variáveis de ambiente e parâmetros utilizados pelo sistema.

### `app.py`

Ponto de entrada principal da aplicação.

Responsável pela interface web desenvolvida com **Streamlit** e pela interação do usuário com o chatbot.

### `raw/`

Pasta que contém os documentos utilizados como base de conhecimento pelo sistema RAG.

---

# Tecnologias utilizadas

O projeto utiliza principalmente:

* **Python** — linguagem principal da aplicação;
* **Streamlit** — desenvolvimento da interface web;
* **LangChain** — integração entre recuperação de contexto e modelo de linguagem;
* **OpenRouter** — acesso ao modelo de linguagem;
* **Sentence Transformers** — geração dos embeddings;
* **ChromaDB** — armazenamento e busca vetorial;
* **PyPDF** — leitura e extração de conteúdo dos documentos PDF.

---

# Execução rápida

Depois de possuir **Python**, **Git** e uma **chave da OpenRouter**, execute:

```bash
git clone https://github.com/projects-yuri/VAARChatbot.git
cd VAARChatbot
python -m venv .venv
```

No CMD do Windows:

```cmd
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Crie o arquivo `.env` na raiz do projeto:

```env
OPENROUTER_API_KEY=SUA_CHAVE_OPENROUTER
OPENROUTER_MODEL=z-ai/glm-5.2:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
TOP_K=5
```

Depois, crie o índice:

```bash
python -m src.ingestion --reset
```

E inicie a aplicação:

```bash
streamlit run app.py
```

Por fim, abra no navegador:

```text
http://localhost:8501
```

---

## Observações

Na primeira execução, a geração dos embeddings pode levar mais tempo devido ao carregamento do modelo e ao processamento dos documentos.

Após o índice vetorial ser criado, o sistema poderá reutilizar os dados armazenados no ChromaDB para realizar novas consultas.

Cada usuário deve utilizar sua **própria chave da OpenRouter** no arquivo `.env`.

O arquivo `.env` não deve ser publicado no repositório.
