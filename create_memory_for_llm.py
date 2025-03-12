import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from load_postgres_data import load_postgres_data

# For images
import pytesseract
from PIL import Image

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\arun.cm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

PDF_DATA_PATH = "data"              # Folder containing PDFs
IMAGE_DATA_PATH = "images"          # Folder containing images
DB_FAISS_PATH = "vectorstore/db_faiss"

# PostgreSQL connection details
HOST = "127.0.0.1"
PORT = 5432
DBNAME = "travel_rag"
USER = "postgres"
PASSWORD = "experion@123"

# SQL query (customize as needed)
SQL_QUERY = "SELECT name, department, email, salary, join_date FROM employees;"


# ------------------------------------------------------------
# 2. Functions
# ------------------------------------------------------------

def load_pdf_files(data_path: str):
    """
    Loads all PDFs from the given folder and returns them as Document objects.
    """
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

def load_db_data(host, port, dbname, user, password, query):
    """
    Loads data from a PostgreSQL table or query and returns a list of Document objects.
    """
    return load_postgres_data(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        query=query
    )

def load_images_and_extract_text(image_dir: str):
    """
    Loads all images from a directory and extracts text using Tesseract OCR.
    Returns a list of Document objects.
    """
    image_docs = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            path = os.path.join(image_dir, filename)
            try:
                text = pytesseract.image_to_string(Image.open(path))
                if text.strip():
                    image_docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": path}
                        )
                    )
            except Exception as e:
                print(f"Warning: Could not process image {path} due to {e}")
    return image_docs

def create_chunks(documents):
    """
    Splits each document's text into manageable chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

def get_text_embedding_model():
    """
    Returns a HuggingFace-based text embedding model.
    Here we use the all-MiniLM-L6-v2 model.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ------------------------------------------------------------
# 3. Main Workflow
# ------------------------------------------------------------

# 3a. Load PDF documents
pdf_documents = load_pdf_files(PDF_DATA_PATH)
print(f"Loaded {len(pdf_documents)} PDF page(s).")

# 3b. Load data from PostgreSQL
db_documents = load_db_data(HOST, PORT, DBNAME, USER, PASSWORD, SQL_QUERY)
print(f"Loaded {len(db_documents)} database record(s).")

# 3c. Load images and extract text
image_documents = load_images_and_extract_text(IMAGE_DATA_PATH)
print(f"Loaded {len(image_documents)} image document(s).")

# Combine all documents (PDF + database + images)
all_documents = pdf_documents + db_documents + image_documents
print(f"Total combined documents: {len(all_documents)}")

# 3d. Split into chunks
text_chunks = create_chunks(all_documents)
print(f"Total text chunks after splitting: {len(text_chunks)}")

# 3e. Embed and store in FAISS
embedding_model = get_text_embedding_model()
db_faiss = FAISS.from_documents(text_chunks, embedding_model)
db_faiss.save_local(DB_FAISS_PATH)

print("FAISS vectorstore saved successfully at:", DB_FAISS_PATH)
