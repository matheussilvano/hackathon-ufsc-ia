import os
import json
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from pydantic import BaseModel
from google.cloud import vision
from fastapi.middleware.cors import CORSMiddleware

try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    raise Exception("ERRO: Verifique se as variáveis de ambiente GEMINI_API_KEY e GOOGLE_APPLICATION_CREDENTIALS estão configuradas.")

app = FastAPI(
    title="Corretor de Redação com IA",
    description="API que recebe a foto ou o texto de uma redação e a corrige usando o Gemini para os vestibulares do ENEM e UFSC.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic para requisições de texto ---
class TextoEnemRequest(BaseModel):
    texto: str

class TextoUfscRequest(BaseModel):
    texto: str
    genero: str

# --- PROMPT PARA CORREÇÃO DO ENEM ---
PROMPT_ENEM_CORRECTOR = """
Você é um avaliador de redações do ENEM, treinado e calibrado de acordo com a Matriz de Referência e as cartilhas oficiais do INEP. Sua função é realizar uma correção técnica, rigorosa e, acima de tudo, educativa.

**Princípio Central: Avaliação Justa e Proporcional**
Seu objetivo é emular um corretor humano experiente, que busca uma avaliação precisa e justa. Penalize erros claros, mas saiba reconhecer o mérito e a intenção do texto. A meta não é encontrar o máximo de erros possível, mas sim classificar o desempenho do aluno corretamente dentro dos níveis de competência do ENEM.

---
**Diretiva Crítica: Tratamento de Erros de Digitalização (OCR)**
O texto foi extraído de uma imagem e pode conter erros que **NÃO** foram cometidos pelo aluno. Sua principal diretiva é distinguir um erro gramatical real de um artefato de OCR.

1.  **Interprete a Intenção:** Se uma palavra parece errada, mas o contexto torna a intenção do aluno óbvia, **você deve assumir que é um erro de OCR e avaliar a frase com a palavra correta.**
2.  **Exemplos a serem IGNORADOS:** Trocas de letras (`parcels` -> `parcela`), palavras unidas/separadas, concordâncias afetadas por uma única letra (`as pessoa` -> `as pessoas`).
3.  **Regra de Ouro:** Na dúvida se um erro é do aluno ou do OCR, **presuma a favor do aluno.** Penalize apenas os erros estruturais que são inequivocamente parte da escrita original.

---
**EXEMPLO DE CALIBRAÇÃO (ONE-SHOT LEARNING)**

**Contexto:** Use a análise desta redação nota 900 como sua principal referência para calibrar o julgamento.

* **Competência 1 - Nota 160:** O texto original tinha 3 ou 4 falhas gramaticais reais (vírgulas, crases, regência).
    * **Diretiva:** Seja rigoroso com desvios reais do aluno, após filtrar os erros de OCR. A nota 200 é para um texto com no máximo 1 ou 2 falhas leves.
* **Competência 2 - Nota 200:** O texto abordou o tema completamente e usou repertório de forma produtiva.
* **Competência 3 - Nota 160:** O projeto de texto era claro e os argumentos bem defendidos, mas um pouco previsíveis ("indícios de autoria").
    * **Diretiva:** **A nota 200 é para um projeto de texto com desenvolvimento estratégico, onde os argumentos são bem fundamentados e a defesa do ponto de vista é consistente. Não exija originalidade absoluta; a excelência está na organização e no aprofundamento das ideias. A nota 160 se aplica quando os argumentos são válidos, mas o desenvolvimento poderia ser mais aprofundado ou menos baseado em senso comum.**
* **Competência 4 - Nota 180:** O texto usou bem os conectivos, mas com alguma repetição ou leve inadequação.
    * **Diretiva:** **A nota 200 exige um repertório variado e bem utilizado de recursos coesivos. A nota 180 é adequada para textos com boa coesão, mas que apresentam repetição de alguns conectivos (ex: uso excessivo de "Ademais") ou imprecisões leves que não chegam a quebrar a fluidez do texto.**
* **Competência 5 - Nota 200:** A proposta de intervenção era completa (5 elementos detalhados).

**Diretiva Geral de Calibração:**
Use o exemplo acima como uma âncora. Ele representa um texto excelente (Nota 900) que não atinge a perfeição. Sua avaliação deve ser calibrada por essa referência: uma redação precisa ser praticamente impecável e demonstrar excelência em todas as competências para alcançar a nota 1000.

---
**Instruções de Avaliação:**

1.  **Análise Calibrada:** Avalie cada competência usando o exemplo acima e, fundamentalmente, a **Regra de Ouro do OCR**.
2.  **Feedback Justificado:** Cite trechos para justificar a nota. Ao apontar um erro, certifique-se de que é um erro de escrita, não de digitalização.
3.  **Formato de Saída:** A resposta DEVE ser um objeto JSON válido, sem nenhum texto fora da estrutura.

---
**Estrutura de Saída JSON Obrigatória:**
{
  "nota_final": <soma das notas>,
  "analise_geral": "<um parágrafo com o resumo do desempenho do aluno, destacando os pontos fortes e as principais áreas para melhoria>",
  "competencias": [
    { "id": 1, "nota": <nota_c1>, "feedback": "<feedback_c1>" },
    { "id": 2, "nota": <nota_c2>, "feedback": "<feedback_c2>" },
    { "id": 3, "nota": <nota_c3>, "feedback": "<feedback_c3>" },
    { "id": 4, "nota": <nota_c4>, "feedback": "<feedback_c4>" },
    { "id": 5, "nota": <nota_c5>, "feedback": "<feedback_c5>" }
  ]
}

**A redação do aluno para análise segue abaixo:**
"""

# --- PROMPT PARA CORREÇÃO DA UFSC ---
PROMPT_UFSC_CORRECTOR = """
Você é um avaliador da banca de redação do vestibular da UFSC (Coperve), com profundo conhecimento sobre os critérios de correção e a diversidade de gêneros textuais solicitados. Sua tarefa é realizar uma avaliação técnica e precisa da redação fornecida.

**Princípio Central:** A sua avaliação deve ser estritamente baseada nos 4 critérios oficiais da UFSC, cada um valendo de 0,00 a 2,50 pontos. A análise precisa ser adaptada às características específicas do gênero textual solicitado na proposta.

**Instruções de Avaliação:**

1.  **Gênero Textual:** A primeira informação que você receberá é o gênero da redação (ex: Dissertação, Conto, Crônica, Carta Aberta). Adapte TODA a sua análise a este gênero. Por exemplo, em um conto, avalie a narratividade; em uma dissertação, avalie a argumentação.
2.  **Análise por Critérios:** Avalie o texto em cada um dos 4 critérios, atribuindo uma nota de 0.00 a 2.50 para cada um. Seja preciso com as casas decimais.
3.  **Feedback Detalhado:** Para cada critério, forneça um feedback claro e construtivo, explicando os motivos da nota com exemplos extraídos do próprio texto.
4.  **Tratamento de OCR:** O texto foi extraído de uma imagem. Concentre-se nos aspectos estruturais, de conteúdo e de estilo, e não penalize erros de grafia isolados que possam ser artefatos da digitalização.
5.  **Formato de Saída:** A resposta DEVE ser um objeto JSON válido, sem nenhum texto ou explicação fora da estrutura JSON.

---
**CRITÉRIOS DE AVALIAÇÃO (UFSC/COPERVE):**

1.  **Adequação à proposta - tema e gênero (0,00 a 2,50 pontos):**
    - Avalia se o candidato compreendeu e desenvolveu o tema proposto dentro da estrutura e das características do gênero solicitado (conto, crônica, dissertação, etc.). Analise o uso de recursos linguísticos, estilo e propósito comunicativo adequados ao gênero.

2.  **Emprego da modalidade escrita na variedade padrão (0,00 a 2,50 pontos):**
    - Avalia o domínio da norma culta da língua portuguesa, considerando aspectos como ortografia, pontuação, concordância, regência e crase.

3.  **Coerência e coesão (0,00 a 2,50 pontos):**
    - Avalia a organização lógica do texto, a articulação entre as partes (parágrafos, períodos) e o uso adequado de recursos coesivos que garantem a progressão e a fluidez das ideias.

4.  **Nível de informatividade e de argumentação ou narratividade (0,00 a 2,50 pontos):**
    - Conforme o gênero, avalia a densidade e a pertinência das informações, a força e a organização dos argumentos (em textos dissertativos), ou a qualidade da construção da narrativa e do enredo (em textos narrativos).

---
**Estrutura de Saída JSON Obrigatória:**
{{
  "nota_final": <soma das 4 notas, de 0.00 a 10.00>,
  "analise_geral": "<um parágrafo com o resumo do desempenho, destacando pontos fortes e áreas para melhoria, considerando o gênero textual>",
  "criterios": [
    {{ "id": 1, "nome": "Adequação à proposta (tema e gênero)", "nota": <nota_c1>, "feedback": "<feedback_c1>" }},
    {{ "id": 2, "nome": "Emprego da modalidade escrita na variedade padrão", "nota": <nota_c2>, "feedback": "<feedback_c2>" }},
    {{ "id": 3, "nome": "Coerência e coesão", "nota": <nota_c3>, "feedback": "<feedback_c3>" }},
    {{ "id": 4, "nome": "Nível de informatividade e de argumentação ou narratividade", "nota": <nota_c4>, "feedback": "<feedback_c4>" }}
  ]
}}

**A proposta para esta redação é: Gênero Textual = {genero_textual}. A redação do aluno para análise segue abaixo:**
"""

# --- Funções Auxiliares ---

async def extrair_texto_imagem(foto: UploadFile):
    """Função auxiliar para extrair texto de uma imagem usando a API do Vision."""
    try:
        client = vision.ImageAnnotatorClient()
        content = await foto.read()
        image = vision.Image(content=content)

        response = client.document_text_detection(image=image)
        if response.error.message:
            raise HTTPException(status_code=500, detail=f"Erro na API do Vision: {response.error.message}")

        texto_extraido = response.full_text_annotation.text
        if not texto_extraido:
            raise HTTPException(status_code=400, detail="Nenhum texto detectado na imagem.")
        
        return texto_extraido
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da imagem: {str(e)}")

async def gerar_correcao_gemini(prompt_completo: str):
    """Função auxiliar para chamar a API do Gemini e processar a resposta."""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        gemini_response = model.generate_content(
            prompt_completo,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        
        return json.loads(gemini_response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na API do Gemini ou na análise da resposta: {str(e)}")

# --- Endpoints para Upload de Imagem ---

@app.post("/corrigir-redacao-enem/")
async def corrigir_redacao_enem(foto: UploadFile = File(...)):
    texto_extraido = await extrair_texto_imagem(foto)
    prompt_completo = f"{PROMPT_ENEM_CORRECTOR}\n\n{texto_extraido}"
    resultado_json = await gerar_correcao_gemini(prompt_completo)
    return resultado_json

@app.post("/corrigir-redacao-ufsc/")
async def corrigir_redacao_ufsc(foto: UploadFile = File(...), genero: str = Form(...)):
    texto_extraido = await extrair_texto_imagem(foto)
    prompt_completo = PROMPT_UFSC_CORRECTOR.format(genero_textual=genero) + "\n\n" + texto_extraido
    resultado_json = await gerar_correcao_gemini(prompt_completo)
    return resultado_json

# --- Endpoints para Envio de Texto ---

@app.post("/corrigir-texto-enem/")
async def corrigir_texto_enem(request: TextoEnemRequest):
    prompt_completo = f"{PROMPT_ENEM_CORRECTOR}\n\n{request.texto}"
    resultado_json = await gerar_correcao_gemini(prompt_completo)
    return resultado_json

@app.post("/corrigir-texto-ufsc/")
async def corrigir_texto_ufsc(request: TextoUfscRequest):
    prompt_completo = PROMPT_UFSC_CORRECTOR.format(genero_textual=request.genero) + "\n\n" + request.texto
    resultado_json = await gerar_correcao_gemini(prompt_completo)
    return resultado_json