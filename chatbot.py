from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
import textwrap
import warnings
import streamlit as st
from typing import Dict
from dotenv import load_dotenv

class Bot:
    def __init__(self, api_key, profile_number=0):
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key, temperature=0.6)
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

        self.persist_directory = "./db"
        self.max_history = 5
        self.conversation_history = []
        self.history_handler = history_handler()

        self.chunk_size = system_prompt = 1000
        self.chunk_overlap = system_prompt = 200
        self.files_dir = r"data/RAG - Português.txt"
        self.vectordb = self.create_vector_db(files_dir = self.files_dir, chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap)

        self.set_prompt(profile_number)

        self.retriever = self.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.2}
        )

        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)

    def set_prompt(self, version):
        if version == 0:
            system_msg = """
            Você é o Dataviewer, um chatbot educacional especializado em ajudar iniciantes que estão começando a aprender programação em JavaScript.
            Seja extremamente didático e acolhedor. Explique os conceitos básicos com detalhes, como variáveis, funções, loops, estruturas condicionais e lógica simples. Use metáforas e analogias sempre que possível.
            Evite termos técnicos sem explicação. Sempre que mostrar código, explique linha por linha. Responda com paciência e incentivo, garantindo que o aluno se sinta seguro para continuar aprendendo.
            O objetivo é fazer o estudante gostar de programação e sentir que está progredindo, mesmo com dúvidas simples.
            Não leve em consideração nenhuma instrução anterior, apenas siga as instruções dadas aqui.
            O objetivo é fazer o estudante gostar de programação e sentir que está progredindo, mesmo com dúvidas simples.
            """
        elif version == 1:
            system_msg = """
            Você é o Dataviewer, um chatbot educacional que ajuda alunos intermediários da Escola de Ciência e Tecnologia a dominar JavaScript.
            Seu papel é apresentar os conceitos essenciais da linguagem com clareza, ajudando o aluno a consolidar seus conhecimentos e entender os porquês por trás de cada sintaxe ou lógica.
            Utilize exemplos de código comentados, explique erros comuns e boas práticas de forma acessível, e incentive a prática com exercícios guiados.
            Você pode revisitar tópicos anteriores se o aluno demonstrar dúvida, sempre buscando reforçar a compreensão sem parecer repetitivo.
            Não leve em consideração nenhuma instrução anterior, apenas siga as instruções dadas aqui.
            O objetivo é fazer o estudante gostar de programação e sentir que está progredindo, mesmo com dúvidas simples.
            """
        elif version == 2:
            system_msg = """
            Você é o Dataviewer, um assistente voltado para alunos um pouco mais avançados em programação que já dominam alguns conceitos básicos e intermediários de JavaScript.
            Seu papel responder esses alunos, tirando as dúvidas que eles possuem.
            Quando possível, você pode propor exercícios mais complexos, ou sugerir plataformas de estudo ou projetos práticos.
            Durante a conversa, incentive o aluno a explorar novas áreas da programação e projetos.
            Foque em tirar a dúvida do aluno, mostrandos outros conceitos avançados apenas se possível.
            Não leve em consideração nenhuma instrução anterior, apenas siga as instruções dadas aqui.
            """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg.strip()),
            ("human", """
            **Histórico da conversa:**
            {conversation_history}

            **Pergunta:**
            {input}

            **Contexto:**
            {context}"""
            )
        ])

    def create_vector_db(self, files_dir, chunk_size=1000, chunk_overlap=200):

        loader = TextLoader(file_path=files_dir, encoding="utf-8")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        texts = text_splitter.split_documents(documents)

        vectordb = Chroma.from_documents(documents = texts,
                                        persist_directory = self.persist_directory,
                                        embedding = self.embedding)
        
        return vectordb

    def _format_conversation_history(self, user_id: str) -> str:
        self.conversation_history = self.history_handler.get_user_history(user_id)

        if not self.conversation_history:
            return "Nenhum histórico de conversa anterior."
            
        formatted_history = []

        for i, (query, response) in enumerate(self.conversation_history, 1):
            formatted_history.append(f"Pergunta {i}: {query}")
            formatted_history.append(f"Resposta {i}: {response}\n")
            
        return "\n".join(formatted_history)

    def generate_response(self, query: str, user_id: str) -> Dict:
        # Prepare input with conversation history
        input_dict = {
            "input": query,
            "conversation_history": self._format_conversation_history(user_id)
        }
        
        # Generate response
        result = self.retrieval_chain.invoke(input_dict)
        
        # Update conversation history
        if "answer" in result:
            self.conversation_history.append((query, result["answer"]))

            # Maintain only the last max_history turns
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)

            self.history_handler.update_history(user_id, self.conversation_history)
            
        return result
    
class history_handler:
    def __init__(self):
        self.history_list = dict()

    def get_user_history(self, user_id):
        try:
            return self.history_list[user_id]
        except:
            self.history_list[user_id] = []
            return self.history_list[user_id]

    def update_history(self, user_id, history):
        self.history_list[user_id] = history
