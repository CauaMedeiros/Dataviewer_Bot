from chatbot import Bot
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import os

# Define página com largura total
st.set_page_config(layout="wide")

# Define o estado inicial da sessão
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "home"

# Função para mudar a tela
def change_screen(screen_name):
    st.session_state.current_screen = screen_name
    st.rerun()

if st.session_state.current_screen == "home":
    # Divide a página em 3 colunas
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 2, 1, 2, 1, 2, 1])

    # Exibe o título na coluna do meio
    with col4:
        st.image(r"imgs/dataviewer_logo.svg", width=340)

    # Espaçamento
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 2, 1, 2, 1, 2, 1])

    with col2:
        st.image(r"imgs/profile1.svg", width='stretch')
        if st.button("Perfil 1", width='stretch'):
            st.session_state.user_profile_number = 0
            change_screen("chat")
        st.markdown("<p style='text-align: center; color: #8e8f94; font-size:14px'>Didático e Focado em Conceitos Fundamentais</p>", unsafe_allow_html=True)
    with col4:
        st.image(r"imgs/profile2.svg", width='stretch')
        if st.button("Perfil 2", width='stretch'):
            st.session_state.user_profile_number = 1
            change_screen("chat")
        st.markdown("<p style='text-align: center; color: #8e8f94; font-size:14px'>Auxilia a Solucionar as Principais Dúvidas de Programação</p>", unsafe_allow_html=True)
    with col6:
        st.image(r"imgs/profile3.svg", width='stretch')
        if st.button("Perfil 3", width='stretch'):
            st.session_state.user_profile_number = 2
            change_screen("chat")
        st.markdown("<p style='text-align: center; color: #8e8f94; font-size:14px'>Desafia e Apresenta Conceitos Avançados</p>", unsafe_allow_html=True)

elif st.session_state.current_screen == "chat":
    st.sidebar.image(r"imgs/dataviewer_logo.svg", width='stretch')


    with st.sidebar:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #8e8f94; font-size:14px'>Bot Educacional desenvolvido para apoiar alunos com dúvidas em programação. É o resultado de um projeto de pesquisa desenvolvido por estudantes da Escola de Ciência e Tecnologia da UFRN, com o objetivo de facilitar o aprendizado e promover o desenvolvimento de habilidades na área.</p>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True) 
            st.markdown("<h4 style='text-align: center; color: #8e8f94'>INPACTA 2026</h4>", unsafe_allow_html=True)

    if "user_id" not in st.session_state:
        user_id = st.text_input("Digite sua matrícula:")
        if user_id:
            st.session_state["user_id"] = user_id 
            st.rerun()
    else:

        if "bot" not in st.session_state:
            api_key = st.secrets["GOOGLE_API_KEY"]
            profile_number = 0  # ou outro valor se quiser personalizar
            profile_number = st.session_state.user_profile_number
            st.session_state.bot = Bot(api_key, profile_number)

        bot = st.session_state.bot
        user_id = st.session_state.user_id
        
        # Exibir o chat
        if "message_log" not in st.session_state:
            st.session_state.message_log = [{"role": "ai", "content": "Olá. Sou o Dataviewer, qual a sua dúvida hoje? 💻"}]

        # Container para mensagens
        container = st.container()

        with container:
            for message in st.session_state.message_log:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Campo de entrada para o chat
        query = st.chat_input("Digite sua dúvida aqui...")

        if query:
            # Adicionar mensagem do usuário ao log
            st.session_state.message_log.append({"role": "user", "content": query})

            # Gerar resposta do bot
            with st.spinner("Processando..."):
                response = bot.generate_response(query, user_id)

            # Adicionar resposta do bot ao log
            st.session_state.message_log.append({"role": "ai", "content": response["answer"]})

            st.rerun()