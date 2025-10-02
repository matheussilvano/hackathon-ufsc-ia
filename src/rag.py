import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Importações do Langchain e Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Configuração da API Key do Google ---
try:
    api_key = os.environ["GOOGLE_API_KEY"]
except KeyError:
    # Se a variável de ambiente não estiver definida, pare a execução.
    raise EnvironmentError("A variável de ambiente GOOGLE_API_KEY não foi encontrada. Configure sua API key.")

# --- Constantes e Configurações Globais ---
CHROMA_PERSIST_DIR = "chroma_db"
TEMP_UPLOAD_DIR = "temp_uploads"

# --- Inicialização do FastAPI ---
app = FastAPI(
    title="API de RAG com Gemini",
    description="Faça upload de documentos e faça perguntas sobre eles.",
    version="1.0.0"
)

# --- Modelos de Dados (Pydantic) para Request/Response ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    
class UploadResponse(BaseModel):
    message: str
    filename: str

# --- Lógica da API ---

def process_and_store_document(filepath: str, original_filename: str):
    """Carrega, processa e armazena um documento no ChromaDB."""
    
    # Escolhe o loader correto com base na extensão
    if original_filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif original_filename.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(filepath)
    else:
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado. Use PDF ou PPTX.")

    documents = loader.load()
    
    # Divide o texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Cria embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Conecta ao ChromaDB e adiciona os documentos
    # Se a coleção já existir, ele adiciona os novos documentos a ela.
    vector_store = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory=CHROMA_PERSIST_DIR
    )
    vector_store.persist()

# --- Endpoints da API ---

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint para fazer upload de um arquivo (PDF ou PPTX).
    O arquivo é processado e seus dados são armazenados no banco vetorial.
    """
    if not os.path.exists(TEMP_UPLOAD_DIR):
        os.makedirs(TEMP_UPLOAD_DIR)
        
    temp_filepath = os.path.join(TEMP_UPLOAD_DIR, file.filename)
    
    try:
        # Salva o arquivo temporariamente
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Processa e armazena o documento
        process_and_store_document(temp_filepath, file.filename)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo: {str(e)}")
    finally:
        # Limpa o arquivo temporário
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
    return {"message": "Arquivo processado e armazenado com sucesso.", "filename": file.filename}


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Endpoint para fazer uma pergunta sobre os documentos já enviados.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
         raise HTTPException(status_code=404, detail="Nenhum documento foi enviado ainda. Faça o upload primeiro.")

    # Inicializa os modelos
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    # Carrega o banco de vetores persistente
    vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    
    # Cria o retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Busca os 5 chunks mais relevantes
    
    # Cria o prompt template
    prompt_template = ChatPromptTemplate.from_template("""
    Sua tarefa é responder à pergunta do usuário baseando-se estritamente no contexto fornecido.
    Se a resposta não estiver no contexto, diga que não encontrou a informação no documento.

    Contexto:
    {context}

    Pergunta do Usuário:
    {input}

    Resposta concisa e direta:
    """)
    
    # Cria a chain RAG
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    response = document_chain.invoke({
        "input": request.question,
        "context": retriever.get_relevant_documents(request.question)
    })
    
    return {"answer": response}