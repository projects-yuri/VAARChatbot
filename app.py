import streamlit as st

from src.rag import answer

st.set_page_config(page_title="PI-5 • RAG", page_icon="📚")
st.title("📚 Chatbot PI-5 — Fundeb, VAAR e ICMS Educacional")
st.caption("Respostas ancoradas na documentação local indexada.")

question = st.chat_input("Faça uma pergunta sobre os documentos do PI-5")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Consultando os documentos..."):
            try:
                response, sources = answer(question)
                st.write(response)
                with st.expander("Trechos recuperados"):
                    for item in sources:
                        st.markdown(f"**{item['source']} — p. {item['page']}**")
                        st.write(item["text"])
            except Exception as error:
                st.error(str(error))
