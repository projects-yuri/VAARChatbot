# VAARChatbot

Chatbot com arquitetura RAG (Retrieval-Augmented Generation) desenvolvido para responder perguntas com base na documentação do Projeto PI-5, com foco em Fundeb, VAAR e ICMS Educacional do Rio Grande do Sul.
O sistema consulta documentos locais, recupera os trechos mais relevantes e utiliza um modelo de linguagem através da API OpenRouter para gerar respostas fundamentadas nas fontes.

---

## Requisitos

Para executar o projeto localmente é necessário possuir:
- Python 3.11 ou superior
- Git
- Conexão com a internet
- Uma chave de API da OpenRouter
  
As principais bibliotecas utilizadas são:
- Streamlit
- ChromaDB
- LangChain
- LangChain OpenAI
- Sentence Transformers
- PyPDF
- Torch
- Python Dotenv

Todas as dependências estão disponíveis no arquivo:

```text
requirements.txt
