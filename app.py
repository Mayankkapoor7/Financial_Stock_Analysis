import os
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found! Please set it in your environment or .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Constants
FOLDER_PATH = r"C:\Users\mayan\OneDrive\Documents\1-Natural Language Processing\GEN_AI\Langchain\Financial Stock Analysis\articles"
CHROMA_PERSIST_DIR = "./chroma_db"

stock_symbols = ['MSFT', 'NVDA', 'GOOG', 'META', 'AAPL', 'TSM']

@st.cache_resource(show_spinner=False)
def load_and_split_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".html") or filename.endswith(".htm"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    documents.append(Document(page_content=text, metadata={"source": filename}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            split_docs.append(Document(page_content=chunk, metadata={"source": doc.metadata["source"], "chunk": i}))
    return split_docs

@st.cache_resource(show_spinner=False)
def create_or_load_vectorstore(_documents):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        st.info("Loaded existing vectorstore from disk.")
    else:
        vectorstore = Chroma.from_documents(_documents, embeddings, persist_directory=CHROMA_PERSIST_DIR)
        st.success("Created and persisted new vectorstore.")
    return vectorstore

st.title("📊 Financial Stock Analysis Q&A")

# Show available stock symbols in a small info box
st.info(f"Available stock symbols: {', '.join(stock_symbols)}")

with st.spinner("Loading and indexing documents..."):
    docs = load_and_split_documents(FOLDER_PATH)
    vectorstore = create_or_load_vectorstore(docs)

retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

option = st.selectbox("Select report type:", ["Individual Stock Analysis", "Competitor Analysis"])

if option == "Individual Stock Analysis":
    symbol = st.text_input("Enter stock symbol:")
    if symbol:
        query = f"Provide a detailed financial analysis and outlook for the stock {symbol}. Include risks and future prospects."
        with st.spinner("Generating report..."):
            answer = qa_chain.run(query)
        st.markdown("### Analysis Report")
        st.write(answer)

else:  # Competitor Analysis
    symbol1 = st.text_input("Enter first stock symbol:")
    symbol2 = st.text_input("Enter second stock symbol:")
    if symbol1 and symbol2:
        query = f"Compare the financial performance and market position of {symbol1} and {symbol2}. Highlight strengths, weaknesses, and competitive advantages."
        with st.spinner("Generating competitor analysis..."):
            answer = qa_chain.run(query)
        st.markdown("### Competitor Analysis Report")
        st.write(answer)
