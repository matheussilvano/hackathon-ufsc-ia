# Cognita Suite - Hackathon AI Challenge SECCOM UFSC

![Status](https://img.shields.io/badge/status-concluído-brightgreen)
![Evento](https://img.shields.io/badge/evento-Hackathon%20AI%20Challenge-blue)
![Trilha](https://img.shields.io/badge/trilha-Educação-9cf)
![Frontend](https://img.shields.io/badge/frontend-HTML%20|%20CSS%20|%20JS-orange)
![Backend](https://img.shields.io/badge/backend-FastAPI-darkgreen)
![IA](https://img.shields.io/badge/IA-Google%20Gemini%20|%20LangChain-purple)
![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-blueviolet)

O **Cognita Suite** é uma aplicação web desenvolvida para a trilha de Educação do **Hackathon AI Challenge da SECCOM UFSC**. A suíte visa potencializar os estudos de alunos, oferecendo ferramentas inteligentes para correção de redações e consulta a materiais de estudo, tudo impulsionado por modelos de linguagem de ponta.

<img width="1367" height="677" alt="image" src="https://github.com/user-attachments/assets/3eafcab0-9556-462a-800d-7b021ce6c3b3" />


## 📚 Sumário
1. [Sobre o Projeto](#1-sobre-o-projeto)
2. [Módulos da Suíte](#2-módulos-da-suíte)
    - [Módulo Grifo (Corretor de Redações)](#grifo---corretor-de-redações)
    - [Módulo Sinapse (Base de Conhecimento com RAG)](#sinapse---base-de-conhecimento-com-rag)
3. [🧠 Detalhes da Implementação de IA](#3-detalhes-da-implementação-de-ia)
    - [IA do Grifo: Prompt Engineering e Análise Calibrada](#ia-do-grifo-prompt-engineering-e-análise-calibrada)
    - [IA do Sinapse: Arquitetura RAG com LangChain](#ia-do-sinapse-arquitetura-rag-com-langchain)
4. [🛠️ Stack Tecnológica](#4-stack-tecnológica)
5. [🚀 Como Executar o Projeto](#5-como-executar-o-projeto)
6. [👨‍💻 Autor](#6-autor)

## 1. Sobre o Projeto
A Cognita Suite é composta por dois módulos principais integrados em uma interface web fluida e moderna: **Grifo**, um corretor de redações para os vestibulares do ENEM e da UFSC, e **Sinapse**, um sistema de perguntas e respostas que transforma documentos de estudo em uma base de conhecimento interativa.

A aplicação foi construída com um frontend em HTML, CSS e JavaScript puros, que se comunica com um backend robusto em **FastAPI (Python)**, onde toda a lógica de Inteligência Artificial é processada.

## 2. Módulos da Suíte

### Grifo - Corretor de Redações
**Grifo** é uma ferramenta de IA projetada para analisar e avaliar redações nos moldes dos vestibulares mais importantes de Santa Catarina e do Brasil.

- **Múltiplos Formatos de Entrada:** Aceita o envio de redações tanto por **upload de imagem** (utilizando Google Cloud Vision para OCR) quanto por **texto digitado**.
- **Modelos de Correção:** Oferece correção especializada para o **ENEM**, baseada nas 5 competências, e para a **UFSC**, avaliando os 4 critérios da banca e adaptando-se ao gênero textual solicitado (conto, dissertação, etc.).
- **Feedback Detalhado:** Fornece uma análise geral, nota final e um feedback construtivo para cada critério de avaliação, ajudando o aluno a identificar pontos fortes e fracos.
- **Histórico Local:** Salva as correções no `localStorage` do navegador, permitindo que o usuário acesse seu histórico de desempenho.

<img width="1367" height="677" alt="image" src="https://github.com/user-attachments/assets/f7dce579-a68b-464d-ad7a-9591d82a9a85" />


### Sinapse - Base de Conhecimento com RAG
**Sinapse** é um assistente de estudos que utiliza a técnica de **Retrieval-Augmented Generation (RAG)** para responder perguntas com base em documentos fornecidos pelo usuário.

- **Upload de Documentos:** O usuário pode fazer upload de materiais de estudo nos formatos **PDF** e **PPTX**.
- **Base de Conhecimento Persistente:** Os documentos são processados, vetorizados e armazenados em uma base de dados vetorial **ChromaDB** local, criando uma base de conhecimento duradoura.
- **Consultas em Linguagem Natural:** O aluno pode fazer perguntas sobre o conteúdo dos documentos, e a IA buscará as informações mais relevantes para formular uma resposta precisa e contextualizada.

## 3. Detalhes da Implementação de IA
O coração do projeto está no uso estratégico de modelos generativos da família **Google Gemini**, orquestrados de maneiras distintas para cada módulo.

<img width="1367" height="677" alt="image" src="https://github.com/user-attachments/assets/b28defde-6c23-4719-9e35-8544b7cdb68f" />


### IA do Grifo: Prompt Engineering e Análise Calibrada
Para o módulo de correção de redações, a precisão e a justiça da avaliação eram cruciais. A técnica central utilizada foi o **Prompt Engineering Avançado**.

- **Modelo Utilizado:** `Gemini-2.5-pro`.
- **Persona e Calibragem (One-Shot Learning):** O prompt instrui o modelo a atuar como um "avaliador treinado e calibrado" da banca específica (ENEM ou UFSC). Para o ENEM, o prompt inclui um exemplo completo de uma redação nota 900, com a pontuação detalhada por competência. Isso funciona como um exemplo de *one-shot learning*, ancorando o julgamento do modelo a um padrão de referência realista e de alta qualidade.
- **Diretiva Crítica para OCR:** Uma inovação chave no prompt é a "Diretiva Crítica" que instrui o modelo a **distinguir erros gramaticais reais de artefatos de digitalização (OCR)**. O modelo é orientado a presumir a favor do aluno em caso de dúvida, focando em erros estruturais em vez de penalizar pequenas falhas de extração de texto.
- **Saída Estruturada:** A API do Gemini é chamada com o parâmetro `response_mime_type="application/json"`. Isso força o modelo a gerar sua resposta diretamente no formato JSON especificado no prompt, garantindo uma integração robusta e sem falhas com o frontend, eliminando a necessidade de parsing de texto.

### IA do Sinapse: Arquitetura RAG com LangChain
O módulo Sinapse implementa uma pipeline clássica e eficiente de **Retrieval-Augmented Generation (RAG)**, utilizando o framework **LangChain** para orquestrar todas as etapas.

- **Modelos Utilizados:**
    - **Embeddings:** `models/embedding-001` para transformar os textos em vetores numéricos.
    - **Geração de Resposta:** `Gemini-2.5-pro` para sintetizar a resposta final.
- **Pipeline RAG:**
    1.  **Carregamento e Divisão:** Documentos (PDF, PPTX) são carregados com `PyPDFLoader` ou `UnstructuredPowerPointLoader` e divididos em pedaços menores (*chunks*) com `RecursiveCharacterTextSplitter` para otimizar a busca.
    2.  **Vetorização e Armazenamento:** Cada *chunk* é convertido em um vetor de embeddings pelo modelo `embedding-001` e armazenado no **ChromaDB**. O uso de um cliente persistente (`PersistentClient`) garante que a base de conhecimento não seja perdida ao reiniciar a API.
    3.  **Recuperação (Retrieval):** Quando o usuário faz uma pergunta, ela também é vetorizada. O ChromaDB realiza uma busca por similaridade de cosseno para encontrar os *chunks* de texto mais relevantes para a pergunta.
    4.  **Síntese (Generation):** Os *chunks* recuperados são inseridos em um prompt final, juntamente com a pergunta original. O modelo `Gemini-2.5-pro` recebe esse contexto e é instruído a responder à pergunta baseando-se **estritamente** nas informações fornecidas, evitando alucinações.
- **Orquestração com LangChain:** O LangChain é utilizado para conectar todos esses componentes de forma coesa, desde a criação do retriever (`vector_store.as_retriever()`) até a montagem da cadeia de recuperação final (`create_retrieval_chain`).

## 4. Stack Tecnológica
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Backend:** Python 3.10+, FastAPI
- **Inteligência Artificial:**
    - **Modelos:** Google Gemini 1.5 Pro, Google Embedding-001
    - **Frameworks:** LangChain, Google Generative AI for Python
    - **OCR:** Google Cloud Vision API
- **Banco de Dados Vetorial:** ChromaDB (com persistência local)
- **Servidor (Desenvolvimento):** Uvicorn

## 5. Como Executar o Projeto

Execute o projeto em: https://front-end-hackathon-zeta.vercel.app/

## 6. Autores
Desenvolvido por Matheus Silvano, Bernardo Thomas e Igor do Carmo
