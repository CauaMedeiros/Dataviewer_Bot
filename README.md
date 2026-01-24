# Dataviewer Bot 🤖

Um chatbot educacional inteligente desenvolvido para apoiar alunos com dúvidas em programação JavaScript. O projeto utiliza inteligência artificial generativa e técnicas avançadas de RAG (Retrieval Augmented Generation) para fornecer respostas personalizadas e contextualizadas.

---

## 📋 Características

- **3 Perfis de Usuário Personalizados:**
  - **Perfil 1:** Didático e focado em conceitos fundamentais (para iniciantes)
  - **Perfil 2:** Auxilia na solução das principais dúvidas de programação (intermediário)
  - **Perfil 3:** Desafia e apresenta conceitos avançados (avançado)

- **RAG (Retrieval Augmented Generation):** Respostas baseadas em base de conhecimento curada em português
- **Histórico de Conversa:** Mantém contexto das interações anteriores por usuário
- **Interface Intuitiva:** Desenvolvido com Streamlit para experiência web fluida
- **API Google Generative AI:** Utiliza o modelo Gemini para geração de texto

---

## 🛠️ Tecnologias Utilizadas

- **Frontend:** [Streamlit](https://streamlit.io/) - Framework para aplicações web interativas
- **LLM:** Google Generative AI (Gemini)
- **Framework IA:** [LangChain](https://www.langchain.com/) - Orquestração de componentes IA
- **Vector DB:** [Chroma](https://www.trychroma.com/) - Banco de dados vetorial para RAG
- **Embeddings:** Google Generative AI Embeddings
- **Outros:** pandas, python-dotenv

---

## 📦 Instalação

### Pré-requisitos
- Python 3.8+
- pip (gerenciador de pacotes Python)
- Chave de API do Google Generative AI

### Passos de Instalação

1. **Clone ou baixe o projeto:**
```bash
cd "seu_caminho/Dataviewer_Bot"
```

2. **Crie um ambiente virtual (recomendado):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente:**
Crie um arquivo `.env` na raiz do projeto:
```
GOOGLE_API_KEY=sua_chave_api_aqui
```

Obtena sua chave em: https://ai.google.dev/

---

## 🚀 Como Executar

```bash
streamlit run main.py
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501`

---

## 📁 Estrutura do Projeto

```
Dataviewer_Bot/
├── main.py                 # Arquivo principal com interface Streamlit
├── chatbot.py             # Classe Bot e lógica do chatbot
├── requirements.txt       # Dependências do projeto
├── .env                   # Variáveis de ambiente (não incluír no Git)
├── data/
│   └── RAG - Português.txt  # Base de conhecimento para RAG
├── db/
│   ├── chroma.sqlite3     # Banco de dados Chroma
│   └── [embeddings]/      # Diretório com embeddings armazenados
├── imgs/
│   ├── dataviewer_logo.svg
│   ├── profile1.svg
│   ├── profile2.svg
│   └── profile3.svg
└── README.md              # Este arquivo
```

---

## 🔧 Configuração

### Adicionar Dados de Treinamento

1. Adicione seus documentos em texto ao arquivo `data/RAG - Português.txt`
2. O chatbot automaticamente reconstruirá o banco de dados vetorial na próxima execução

### Modificar Prompts do Sistema

Edite a função `set_prompt()` em [chatbot.py](chatbot.py#L44) para personalizar as instruções dos perfis.

---

## 💬 Como Usar

1. **Inicie a aplicação**
2. **Selecione um dos 3 perfis** de acordo com seu nível de conhecimento
3. **Digite sua matrícula** (ID de usuário)
4. **Faça suas perguntas** sobre programação em JavaScript
5. **O bot responderá** com base no perfil selecionado e no contexto anterior

---

## 📊 Arquitetura

```
Usuário (Streamlit)
    ↓
main.py (Interface)
    ↓
chatbot.py (Classe Bot)
    ↓
LangChain (Orquestração)
    ├── Retriever (Chroma - RAG)
    ├── LLM (Google Generative AI)
    └── Prompt Chain (Processamento)
    ↓
Google Generative AI
    ↓
Resposta personalizada ao usuário
```

**Projeto de Pesquisa**
- Desenvolvido por estudantes da Escola de Ciência e Tecnologia da UFRN
- Apresentado em INPACTA 2026
- Objetivo: Facilitar o aprendizado e promover desenvolvimento de habilidades em programação

---

## 📄 Licença

Este projeto é fornecido como está para fins educacionais.

---

**Última atualização:** Janeiro de 2026
