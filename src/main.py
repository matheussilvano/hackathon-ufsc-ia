import os
import json
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import vision

try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    raise Exception("ERRO: Verifique se as variáveis de ambiente GEMINI_API_KEY e GOOGLE_APPLICATION_CREDENTIALS estão configuradas.")

app = FastAPI(
    title="Corretor de Redação ENEM com IA",
    description="API que recebe a foto de uma redação, extrai o texto e a corrige usando o Gemini.",
    version="1.0.0"
)

PROMPT_ENEM_CORRECTOR = """
Você é um corretor especialista da banca do ENEM. Sua única função é analisar a redação que será fornecida a seguir e avaliá-la estritamente com base nas 5 competências oficiais do ENEM. Você deve ser rigoroso, técnico e educativo.

Para cada uma das 5 competências, avalie a redação e forneça:
1. Uma nota de 0 a 200 (em múltiplos de 40).
2. Um feedback conciso e específico, explicando o porquê da nota com exemplos do próprio texto.
3. Desconsidere erros de ortografia que não afetem a compreensão do texto e que ocorrem pela grafia da pessoa.
4. Responda de forma clara, sem palavras rebuscadas ou jargões técnicos, para melhor compreensão do aluno.

As competências são:
- Competência 1: Domínio da escrita formal da língua portuguesa (É avaliado se a redação do participante está adequada às regras de ortografia, como acentuação, ortografia, uso de hífen, emprego de letras maiúsculas e minúsculas e separação silábica. Ainda são analisadas a regência verbal e nominal, concordância verbal e nominal, pontuação, paralelismo, emprego de pronomes e crase.).
- Competência 2: Compreender o tema e não fugir do que é proposto. (Avalia as habilidades integradas de leitura e de escrita do candidato. O tema constitui o núcleo das ideias sobre as quais a redação deve ser organizada e é caracterizado por ser uma delimitação de um assunto mais abrangente.)
- Competência 3: Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista (O candidato precisa elaborar um texto que apresente, claramente, uma ideia a ser defendida e os argumentos que justifiquem a posição assumida em relação à temática da proposta da redação. Trata da coerência e da plausibilidade entre as ideias apresentadas no texto, o que é garantido pelo planejamento prévio à escrita, ou seja, pela elaboração de um projeto de texto.)
- Competência 4: Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação (São avaliados itens relacionados à estruturação lógica e formal entre as partes da redação. A organização textual exige que as frases e os parágrafos estabeleçam entre si uma relação que garanta uma sequência coerente do texto e a interdependência entre as ideias. Preposições, conjunções, advérbios e locuções adverbiais são responsáveis pela coesão do texto porque estabelecem uma inter-relação entre orações, frases e parágrafos. Cada parágrafo será composto por um ou mais períodos também articulados. Cada ideia nova precisa estabelecer relação com as anteriores.).
- Competência 5: Respeito aos direitos humanos (Apresentar uma proposta de intervenção para o problema abordado que respeite os direitos humanos. Propor uma intervenção para o problema apresentado pelo tema significa sugerir uma iniciativa que busque, mesmo que minimamente, enfrentá-lo. A elaboração de uma proposta de intervenção na prova de redação do Enem representa uma ocasião para que o candidato demonstre o preparo para o exercício da cidadania, para atuar na realidade em consonância com os direitos humanos.)

Sua resposta DEVE ser um objeto JSON válido, sem nenhum texto ou explicação adicional fora do JSON. A estrutura deve ser a seguinte:
{
  "nota_final": <soma das notas>,
  "analise_geral": "<um parágrafo com o resumo do desempenho do aluno>",
  "competencias": [
    { "id": 1, "nota": <nota_c1>, "feedback": "<feedback_c1>" },
    { "id": 2, "nota": <nota_c2>, "feedback": "<feedback_c2>" },
    { "id": 3, "nota": <nota_c3>, "feedback": "<feedback_c3>" },
    { "id": 4, "nota": <nota_c4>, "feedback": "<feedback_c4>" },
    { "id": 5, "nota": <nota_c5>, "feedback": "<feedback_c5>" }
  ]
}

A redação do aluno para análise segue abaixo:
"""

@app.post("/corrigir-redacao/")
async def corrigir_redacao(foto: UploadFile = File(...)):
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da imagem: {str(e)}")

    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt_completo = f"{PROMPT_ENEM_CORRECTOR}\n\n{texto_extraido}"

        generation_config = genai.GenerationConfig(
            temperature=0.1  # Um valor baixo para respostas consistentes
        )
        
        gemini_response = model.generate_content(
            prompt_completo,
            generation_config=generation_config
        )
        
        cleaned_text = gemini_response.text.strip().replace("```json", "").replace("```", "")
        
        resultado_json = json.loads(cleaned_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na API do Gemini ou na análise da resposta: {str(e)}")

    return resultado_json