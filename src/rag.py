import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

# --- Importações do Langchain, ChromaDB e Google ---
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# --- Configuração do Roteador ---
router = APIRouter()

# --- Configuração da API Key ---
try:
    gemini_api_key = os.environ["GEMINI_API_KEY"]
except KeyError:
    raise EnvironmentError("A variável de ambiente GEMINI_API_KEY não foi encontrada. Configure sua API key.")

# --- Constantes e Configurações Globais ---
CHROMA_PERSIST_DIR = "chroma_db_persistent"
CHROMA_COLLECTION_NAME = "ufsc_hackathon_rag"
TEMP_UPLOAD_DIR = "temp_uploads"

# --- Modelos de Dados (Pydantic) ---
class QueryRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    source: str
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]

class UploadResponse(BaseModel):
    message: str
    filename: str

# --- Inicialização do Cliente ChromaDB e Embeddings ---
# Garante que o diretório de persistência exista
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Cliente ChromaDB persistente para garantir que os dados sejam salvos localmente
persistent_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Modelo de embeddings que será usado tanto para armazenar quanto para consultar
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)

# Instância principal do Chroma que será usada pela aplicação
vector_store = Chroma(
    client=persistent_client,
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=embeddings,
)

# --- Lógica da Aplicação ---
def process_and_store_document(filepath: str, original_filename: str):
    """Carrega, processa e armazena o documento no ChromaDB, garantindo que não haja duplicatas."""
    
    # 1. Buscar os documentos existentes com base no nome do arquivo no metadado.
    existing_docs = vector_store.get(where={"source": original_filename})
    
    # 2. Se a busca retornar IDs, significa que o documento já existe e precisa ser removido antes de adicionar a nova versão.
    if existing_docs and existing_docs["ids"]:
        vector_store.delete(ids=existing_docs["ids"])

    # 3. Carregar e processar o novo documento.
    if original_filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif original_filename.endswith((".pptx", ".ppt")):
        loader = UnstructuredPowerPointLoader(filepath)
    else:
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado. Use PDF ou PPTX.")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 4. Adicionar metadados a cada chunk para rastrear a origem.
    for doc in docs:
        doc.metadata = {"source": original_filename}

    # 5. Adicionar os novos chunks processados ao banco de dados.
    vector_store.add_documents(docs)


# --- Endpoints da API de RAG ---
@router.post("/upload", response_model=UploadResponse, summary="Upload de Documento")
async def upload_document(file: UploadFile = File(...)):
    if not os.path.exists(TEMP_UPLOAD_DIR):
        os.makedirs(TEMP_UPLOAD_DIR)

    temp_filepath = os.path.join(TEMP_UPLOAD_DIR, file.filename)

    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        process_and_store_document(temp_filepath, file.filename)

    except Exception as e:
        # Fornece um erro mais detalhado em caso de falha
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo: {str(e)}")
    finally:
        # Garante que o arquivo temporário seja sempre removido
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

    return {"message": "Arquivo processado e armazenado com sucesso.", "filename": file.filename}


@router.post("/query", response_model=QueryResponse, summary="Consulta sobre Documentos")
async def query_documents(request: QueryRequest):
    # Verifica se a coleção no ChromaDB contém documentos
    if vector_store._collection.count() == 0:
         raise HTTPException(status_code=404, detail="Nenhum documento foi enviado ainda. Faça o upload primeiro.")

    # O vector_store já está inicializado, então criamos o retriever diretamente
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Modelo de linguagem para gerar as respostas
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, google_api_key=gemini_api_key)

    # Template do prompt para guiar o modelo
    prompt_template = ChatPromptTemplate.from_template("""
    Persona: Você é um tutor de IA, amigável e didático. Sua única função é ensinar usando estritamente o conteúdo dos documentos fornecidos.

    Regra Principal: Sua única fonte de verdade é o material fornecido no "Contexto". Não use nenhum conhecimento externo. Se a resposta não estiver no material, diga isso claramente.

    Instruções:
    1.  Analise todo o "Contexto" abaixo, que pode conter vários trechos do material.
    2.  Sintetize uma resposta coesa a partir desses trechos.
    3.  Use uma linguagem clara e educativa, como um professor faria.
    4.  Siga OBRIGATORIAMENTE o formato de saída HTML.

    Formato de Saída OBRIGATÓRIO (Apenas HTML):
    Sua resposta deve ser APENAS código HTML. Comece diretamente com a tag <div> e não inclua `<html>` ou `<body>`.
    -   Use `<div class="resposta-tutor">` como contêiner principal.
    -   Use `<h3>` para o título principal da explicação.
    -   Use `<p>` para o texto explicativo.
    -   Use `<ul>` e `<li>` para listas.
    -   Use `<b>` ou `<strong>` para destacar termos importantes.

    Exemplo de Resposta para uma Pergunta:
    <div class="resposta-tutor">
        <h3>O Processo de Mitose</h3>
        <p>Com base no material, a mitose é um processo fundamental de <b>divisão celular</b> que resulta em duas células-filhas geneticamente idênticas.</p>
        <p>As etapas principais são:</p>
        <ul>
            <li><b>Prófase:</b> Os cromossomos se condensam.</li>
            <li><b>Metáfase:</b> Os cromossomos se alinham no centro.</li>
        </ul>
    </div>

    Exemplo de Resposta Quando a Informação Não é Encontrada:
    <div class="resposta-tutor">
        <h3>Informação Não Encontrada</h3>
        <p>Consultei todo o material disponível, mas não encontrei uma resposta para sua pergunta. Por favor, tente reformular a pergunta ou questione sobre um tópico abordado no documento.</p>
    </div>

    Contexto:
    {context}

    Pergunta do Usuário:
    {input}

    Sua resposta em HTML:
    """)

    # Criação da cadeia de documentos e da cadeia de recuperação
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Invocação da cadeia para obter a resposta
    response = retrieval_chain.invoke({"input": request.question})

    sources = []
    if "context" in response and response["context"]:
        for doc in response["context"]:
            # Usamos um set para evitar adicionar a mesma fonte múltiplas vezes
            source_name = doc.metadata.get("source", "Fonte desconhecida")
            if source_name not in {s.source for s in sources}:
                 sources.append(SourceDocument(source=source_name, content=doc.page_content))


    return {"answer": response["answer"], "sources": sources}