import copy
from click import File
from fastapi import FastAPI, File, Query, UploadFile
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, dotenv_values
from langchain.chat_models import init_chat_model
import getpass
import os
from langchain_community.document_loaders import PyPDFLoader

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


llm = init_chat_model("gpt-4o-mini", model_provider="openai")

load_dotenv()  # load environment variables from .env file
config = dotenv_values()  # get all the values as a dictionary

# vectorstore
vectorstore = Chroma(
    persist_directory="./rag-files",
    embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
)

# setup the retriever for the similarity
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.7}
)

TEMPLATE = """
Answer the following question:
{question}

To answer the question, use only the following context:
{context}
"""


app = FastAPI()


@app.get("/ask")
def ask_ib_assistant(q: str = Query(..., description="Your IBS question")):
    prompt_template = PromptTemplate.from_template(TEMPLATE)

    chat = ChatOpenAI(model="gpt-4", model_kwargs={"seed": 365}, max_tokens=250)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | chat
        | StrOutputParser()
    )
    response_answer = chain.invoke(q)
    return {"question": q, "answer": response_answer}


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    # Process the uploaded file and update the vectorstore
    with open(f"./rag-files/{file.filename}", "wb") as f:
        f.write(file.file.read())

    loader_pdf = PyPDFLoader(f"./rag-files/{file.filename}")
    documents = loader_pdf.load()
    pages_pdf_deep_copy = copy.deepcopy(documents)
    for i in range(len(pages_pdf_deep_copy)):
        pages_pdf_deep_copy[i].page_content = " ".join(
            pages_pdf_deep_copy[i].page_content.split()
        )
    char_splitter = CharacterTextSplitter(
        separator=".", chunk_size=500, chunk_overlap=50
    )

    pages_pdf_split = char_splitter.split_documents(pages_pdf_deep_copy)
    vectorstore.add_documents(pages_pdf_split)
    return {"filename": file.filename}
