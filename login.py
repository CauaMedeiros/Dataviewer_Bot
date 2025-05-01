import streamlit as st
import psycopg2
import bcrypt


# Conectar ao PostgreSQL
def conectar_bd():
    return psycopg2.connect(
        dbname="recdatabase",
        user="nomedousuario",
        password="senha",
        host="localhost",
        port="5432"
    )

# Criar usuário no banco de dados
def criar_usuario(nome, matricula, senha):
    conn = conectar_bd()
    cur = conn.cursor()
    
    # Criptografar senha antes de salvar
    senha_hash = bcrypt.hashpw(senha.encode(), bcrypt.gensalt()).decode()
    
    try:
        cur.execute("INSERT INTO usuarios (nome, matricula, senha) VALUES (%s, %s, %s)", (nome, matricula, senha_hash))
        conn.commit()
        st.success("Usuário criado com sucesso!")
    except psycopg2.Error as e:
        st.error(f"Erro ao criar usuário: {e}")
    finally:
        cur.close()
        conn.close()

# Verificar login
def verificar_login(matricula, senha):
    conn = conectar_bd()
    cur = conn.cursor()
    
    cur.execute("SELECT senha FROM usuarios WHERE matricula = %s", (matricula,))
    resultado = cur.fetchone()
    
    cur.close()
    conn.close()
    
    if resultado and bcrypt.checkpw(senha.encode(), resultado[0].encode()):
        return True
    return False

# Interface do Streamlit
st.title("Login")

if 'login' not in st.session_state:
    st.session_state.login = False

menu = st.sidebar.selectbox("Menu", ["Login", "Cadastro"])

if menu == "Cadastro":
    st.subheader("Criar novo usuário")
    nome = st.text_input("Nome")
    matricula = st.text_input("Matricula")
    senha = st.text_input("Senha", type="password")
    
    if st.button("Cadastrar"):
        criar_usuario(nome, matricula, senha)

elif menu == "Login":
    st.subheader("Fazer Login")
    matricula = st.text_input("Matricula")
    senha = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        if verificar_login(matricula, senha):
            st.success("Login realizado com sucesso!")
            st.session_state.login = True
        else:
            st.error("Matricula ou senha incorretos.")

if st.session_state.login:
    print("Login realizado com sucesso!")
    
