from dotenv import load_dotenv
import bs4
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split our documents into chunks of 1000 characters with 200 characters of overlap between chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embd and store all of our document splits in a single command using the Chroma vector store and OpenAIEmbeddings model
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

# Pull a specific prompt format from the Hub
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    """Format the retrieved documents into a single string for the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)


# Define the chain for retrieval-augmented generation (RAG)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Check if this is the main module being run
if __name__ == '__main__':
    # Invoke the chain with a sample question
    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)