import os
import json
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from google.cloud import vision
from fastapi.middleware.cors import CORSMiddleware

try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    raise Exception("ERRO: Verifique se as variáveis de ambiente GEMINI_API_KEY e GOOGLE_APPLICATION_CREDENTIALS estão configuradas.")

app = FastAPI(
    title="Corretor de Redação com IA",
    description="API que recebe a foto de uma redação, extrai o texto e a corrige usando o Gemini para os vestibulares do ENEM e UFSC.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PROMPT PARA CORREÇÃO DO ENEM ---
PROMPT_ENEM_CORRECTOR = """
Você é um avaliador de redações do ENEM, treinado e calibrado de acordo com a Matriz de Referência e as cartilhas oficiais do INEP. Sua função é realizar uma correção técnica, rigorosa e educativa, com um discernimento apurado para diferenciar níveis de excelência.

**Princípio Central:** A sua avaliação deve ser analítica e comparativa. Para cada competência, identifique as características do texto, compare-as com o exemplo de calibração fornecido e, então, enquadre-as no nível de desempenho correspondente da matriz para atribuir a nota. O feedback deve justificar esse enquadramento com exemplos do texto.

---
**EXEMPLO DE CALIBRAÇÃO (ONE-SHOT LEARNING)**

**Contexto:** Você deve usar a seguinte análise de uma redação real nota 900 como sua principal referência para calibrar o seu julgamento, especialmente nas notas mais altas.

**Análise da Correção Oficial (Nota 900):**
* **Competência 1 - Nota 160:** Indica que o texto era muito bom, mas continha 3 ou 4 falhas gramaticais ou de convenções (ex: vírgulas, crases, grafia). **Diretiva:** Seja mais rigoroso na contagem de desvios. A nota 200 é para perfeição virtual (no máximo 2 falhas).
* **Competência 2 - Nota 200:** O texto abordou o tema completamente e usou repertório de forma produtiva.
* **Competência 3 - Nota 160:** Significa que o projeto de texto era claro e os argumentos bem defendidos, mas talvez um pouco previsíveis ou muito baseados em senso comum, caracterizando "indícios de autoria" em vez de uma "autoria" plena e original. **Diretiva:** Diferencie um argumento bem colocado de um argumento verdadeiramente original e perspicaz. Não atribua 200 se a argumentação for apenas correta, mas não excelente.
* **Competência 4 - Nota 180:** Esta nota indica uma performance quase perfeita nos mecanismos de coesão. Provavelmente, o texto usou bem os conectivos, mas com pouca variedade (repetindo "Ademais", "Nesse sentido", etc.) ou com alguma pequena inadequação. **Diretiva:** Avalie não só a presença, mas a **diversidade e a precisão** dos recursos coesivos. A repetição excessiva de conectivos impede a nota 200.
* **Competência 5 - Nota 200:** A proposta de intervenção era completa, com todos os 5 elementos bem detalhados.

**Diretiva Geral de Calibração:** Use este exemplo para ser mais crítico. A nota 1000 é para uma redação excepcional. Uma redação ótima, mas com pequenas falhas, como a do exemplo, deve pontuar entre 880-960.
---

**Instruções de Avaliação:**

1.  **Análise por Níveis Calibrados:** Avalie a redação em cada competência, usando o exemplo acima para refinar sua decisão sobre a nota (0, 40, 80, 120, 160 ou 200).
2.  **Feedback Justificado:** Cite trechos da redação para justificar a nota, explicando por que o texto se enquadra naquele nível (e não em um superior), fazendo referência implícita aos critérios de calibração.
3.  **Tratamento de OCR:** O texto foi extraído de uma imagem. Foque na estrutura e argumentação, e não penalize erros de grafia isolados que possam ser artefatos de digitalização.
4.  **Formato de Saída:** Sua resposta DEVE ser um objeto JSON válido, sem nenhum texto fora da estrutura JSON.

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


@app.post("/corrigir-redacao-enem/")
async def corrigir_redacao_enem(foto: UploadFile = File(...)):
    texto_extraido = await extrair_texto_imagem(foto)

    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt_completo = f"{PROMPT_ENEM_CORRECTOR}\n\n{texto_extraido}"
        
        gemini_response = model.generate_content(
            prompt_completo,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        
        resultado_json = json.loads(gemini_response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na API do Gemini ou na análise da resposta: {str(e)}")

    return resultado_json


@app.post("/corrigir-redacao-ufsc/")
async def corrigir_redacao_ufsc(foto: UploadFile = File(...), genero: str = Form(...)):
    texto_extraido = await extrair_texto_imagem(foto)

    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt_completo = PROMPT_UFSC_CORRECTOR.format(genero_textual=genero) + "\n\n" + texto_extraido

        gemini_response = model.generate_content(
            prompt_completo,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        
        resultado_json = json.loads(gemini_response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na API do Gemini ou na análise da resposta: {str(e)}")

    return resultado_json