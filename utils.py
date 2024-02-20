# Load environment variables from the .env file
# from dotenv import load_dotenv
import os
# load_dotenv()

# Access the API key using the environment variable name
# openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets.OPENAI_API_KEY

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader


from langchain.embeddings import OpenAIEmbeddings
embedding=OpenAIEmbeddings(openai_api_key= openai.api_key)
vectordb = Chroma(
    persist_directory= "C:/Users/USER/Downloads/Retrival_methods/new_chroma_tfgm/db",
    embedding_function=embedding,
    
)


# chroma_retriever = Chroma()
# retriever = vectordb.as_retriever(search_kwargs={"k": 2})
from langchain.chat_models import ChatOpenAI

llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",openai_api_key= openai.api_key)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm= llm,
    retriever= vectordb.as_retriever(),
    return_source_documents=True,
    chain_type="stuff"
   )


def rag_func(question: str) ->str:
    """
    This function takes in user question or prompt and returns a response
    :param: question: string valiue of the question or the prompt from the user
    :returns: string value of the answer to the user question
    """
    response = qa_chain({"query": question})

    return response


def process_llm_response(llm_response):
    result = llm_response['result']
    return result 
    

def s(llm_response):
    for source in llm_response["source_documents"]:
        return  source.metadata['source']


