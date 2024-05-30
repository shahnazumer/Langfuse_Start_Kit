from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import os

os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

os.environ["OPENAI_API_KEY"] = ""

from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler()

# Tests the SDK connection with the server
langfuse_handler.auth_check()

langfuse_handler = CallbackHandler()

urls = [
    "https://raw.githubusercontent.com/langfuse/langfuse-docs/main/public/state_of_the_union.txt",
]
loader = UnstructuredURLLoader(urls=urls)
llm = OpenAI()
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
query = "What did the president say about Ketanji Brown Jackson"
chain = RetrievalQA.from_chain_type(
    llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

chain.invoke(query, config={"callbacks":[langfuse_handler]})