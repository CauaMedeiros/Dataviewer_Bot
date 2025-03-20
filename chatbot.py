import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pysqlite3

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import textwrap
import warnings

class Bot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.6)
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        self.persist_directory = "./db"
        self.max_history = 5
        self.conversation_history = []

        self.vectordb = Chroma(
            collection_name="documents",
            embedding_function=self.embedding,
            persist_directory=self.persist_directory
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
             Você é o Dataviewer, um chatbot que ajuda alunos da Escola de Ciência e Tecnologia com programação em JavaScript.
             Não se apresente no início da conversa.
             Seja didático e objetivo, explicando conceitos com clareza e exemplos práticos.
             Sempre que possível, utilize as informações presentes no documento.
             O aluno está no nível iniciante.
             """),
            ("human", """Histórico da conversa:
            {conversation_history}
            
            Caso a mensagem seja uma pergunta, responda usando o contexto fornecido, se ele for relevante (eu não possuo acesso ao contexto, apenas você possui. caso ele não seja relevante, não precisa ser mencionado) NÃO FALE SOBRE O CONTEXTO NA SUA RESPOSTA. Pergunta: {input}. Contexto: {context}.""")
        ])

        self.retriever = self.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.2}
        )

        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)

    def load_document(self, file_path, chunk_size=1000, chunk_overlap=0):
        loader = TextLoader(file_path=file_path, encoding="utf-8")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        
        text_contents = [doc.page_content for doc in texts]
        self.vectordb.add_texts(texts=text_contents)

    def format_conversation_history(self):
        if not self.conversation_history:
            return "Nenhum histórico de conversa anterior."
        
        formatted_history = []
        for i, (query, response) in enumerate(self.conversation_history, 1):
            formatted_history.append(f"Pergunta {i}: {query}")
            formatted_history.append(f"Resposta {i}: {response}\n")

        return "\n".join(formatted_history)

    def clear_history(self):
        self.conversation_history = []

    def get_conversation_history(self):
        return self.conversation_history.copy()

    def generate_response(self, query):
        input_dict = {
            "input": query,
            "conversation_history": self.format_conversation_history()
        }
        
        result = self.retrieval_chain.invoke(input_dict)
        
        if "answer" in result:
            self.conversation_history.append((query, result["answer"]))
        
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
        
        return result