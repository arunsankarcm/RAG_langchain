from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from load_postgres_data import load_postgres_data

# PostgreSQL connection details
HOST = "127.0.0.1"
PORT = 5432  
DBNAME = "travel_rag"
USER = "postgres"
PASSWORD = "experion@123"

# SQL query (customize as needed)
SQL_QUERY = "SELECT name, department, email, salary, join_date FROM employees;"

def create_memory_for_llm():
    # Load data from PostgreSQL
    documents = load_postgres_data(
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        user=USER,
        password=PASSWORD,
        query=SQL_QUERY
    )

    # Create text chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    # Get embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and save vector store
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return db

if __name__ == "__main__":
    db = create_memory_for_llm()
    print("Vector store created and saved successfully!")