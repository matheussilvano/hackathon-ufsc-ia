from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import redacao, rag

app = FastAPI(
    title="Hackathon UFSC IA API",
    description="API principal que integra os módulos de correção de redação e RAG.",
    version="3.0.0"
)

# --- Middlewares ---
# Adicionando o CORS Middleware para permitir requisições de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Inclusão das Rotas ---
# Incluindo as rotas do módulo de redação
app.include_router(redacao.router, prefix="/redacao", tags=["Redação"])
# Incluindo as rotas do módulo de RAG
app.include_router(rag.router, prefix="/rag", tags=["RAG"])

@app.get("/", summary="Endpoint Raiz", description="Verifica se a API está online.")
def read_root():
    """
    Endpoint raiz para verificar a saúde da API.
    """
    return {"status": "API online!"}