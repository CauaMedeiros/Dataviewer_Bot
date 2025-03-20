import streamlit as st
from chatbot import Bot

st.sidebar.image(r"imgs/dataviewer_full.svg", use_container_width=True)

with st.sidebar:
    st.markdown("<p style='text-align: center; color: #8e8f94; font-size:14px'>Bot Educacional desenvolvido para apoiar alunos com dúvidas em programação. Resultado de um projeto de pesquisa desenvolvido por estudantes da Escola de Ciência e Tecnologia da UFRN, com o objetivo de facilitar o aprendizado e promover o desenvolvimento de habilidades na área.</p>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #8e8f94'>INPACTA 2025</h4>", unsafe_allow_html=True)

api_key = st.secrets["GOOGLE_API_KEY"]
bot = Bot(api_key)

bot.load_document(r"data/RAG - Português.txt")

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Olá. Sou o Dataviewer, qual a sua dúvida hoje? 💻"}]

# container
container = st.container()

# menssagem
with container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
query = st.chat_input("Digite sua dúvida aqui...")

if query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": query})
    
    # Generate AI response
    with st.spinner("Processando..."): 
        response = bot.generate_response(query)
 
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": response["answer"]})
    
    st.rerun()