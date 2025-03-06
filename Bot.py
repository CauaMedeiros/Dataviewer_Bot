from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pysqlite3
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from typing import List, Dict
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from IPython.display import display, Markdown
import textwrap
from chromadb.config import Settings
import warnings
import streamlit as st

st.sidebar.image(r"imgs/dataviewer_full.svg", use_container_width=True)

with st.sidebar:
    st.markdown("<p style='text-align: center; color: #8e8f94; font-size:14px'>Bot Educacional desenvolvido para apoiar alunos com dúvidas em programação. Resultado de um projeto de pesquisa desenvolvido por estudantes da Escola de Ciência e Tecnologia da UFRN, com o objetivo de facilitar o aprendizado e promover o desenvolvimento de habilidades na área.</p>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #8e8f94'>INPACTA 2025</h4>", unsafe_allow_html=True)

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

api_key = st.secrets["GOOGLE_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.6)
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

persist_directory = "./db"
max_history = 5
conversation_history = []

vectordb = Chroma(
            collection_name="documents",
            embedding_function=embedding,
)

prompt = ChatPromptTemplate.from_messages([
            ("system", """
             Você é o Dataviewer, um chatbot que ajuda alunos da Escola de Ciência e Tecnologia com programação em JavaScript.
             Não se apresente no início da conversa.
             Seja didático e objetivo, explicando conceitos com clareza e exemplos práticos.
             No início, apresente-se e explique como pode ajudar.
             Sempre que possível, utilize as informações presentes no documento.
             O aluno está no nível iniciante.
             """),
            ("human", """Histórico da conversa:
            {conversation_history}
            
            Caso a mensagem seja uma pergunta, responda usando o contexto fornecido, se ele for relevante (eu não possuo acesso ao contexto, apenas você possui. caso ele não seja relevante, não precisa ser mencionado) NÃO FALE SOBRE O CONTEXTO NA SUA RESPOSTA. Pergunta: {input}. Contexto: {context}.""")
        ])

retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.2}
        )

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

def load_document(file_path, chunk_size=1000, chunk_overlap=0):
        
        loader = TextLoader(file_path=file_path, encoding="utf-8")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        
        text_contents = [doc.page_content for doc in texts]
        vectordb.add_texts(texts=text_contents)

def format_conversation_history():
    if not conversation_history:
        return "Nenhum histórico de conversa anterior."
        
    formatted_history = []
    for i, (query, response) in enumerate(conversation_history, 1):
        formatted_history.append(f"Pergunta {i}: {query}")
        formatted_history.append(f"Resposta {i}: {response}\n")

    return "\n".join(formatted_history)

def clear_history():
    global conversation_history
    conversation_history = []

def get_conversation_history():
    return conversation_history.copy()

def generate_response(query):
    input_dict = {
        "input": query,
        "conversation_history": format_conversation_history()
    }
    
    result = retrieval_chain.invoke(input_dict)
    
    if "answer" in result:
        conversation_history.append((query, result["answer"]))
    
        if len(conversation_history) > max_history:
            conversation_history.pop(0)
    
    return result

load_document(r"data/RAG - Português.txt")

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
        response = generate_response(query)
 
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": response["answer"]})
    
    st.rerun()