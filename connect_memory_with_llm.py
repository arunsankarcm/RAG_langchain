import os
import psycopg2
from typing import List, Dict, Any

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

# PostgreSQL connection details
HOST = "127.0.0.1"
PORT = 5432
DBNAME = "travel_rag"
USER = "postgres"
PASSWORD = "experion@123"  # Consider using environment variables for passwords

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.01,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

# Function to query database directly
def query_database(user_query):
    """Query the database based on the user's question and return formatted results"""
    # Connect to the database
    conn = psycopg2.connect(
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        user=USER,
        password=PASSWORD
    )
    
    # For this example, we'll just query all employees
    # In a more advanced system, you could parse the user query to determine the SQL query
    sql_query = "SELECT name, department, email, salary, join_date FROM employees;"
    
    cur = conn.cursor()
    cur.execute(sql_query)
    rows = cur.fetchall()
    
    # Get column names
    column_names = [desc[0] for desc in cur.description]
    
    cur.close()
    conn.close()
    
    # Format the database results as text
    result_text = "Database Query Results:\n\n"
    for row in rows:
        row_dict = dict(zip(column_names, row))
        result_text += "\n".join([f"{col}: {val}" for col, val in row_dict.items()])
        result_text += "\n\n"
    
    return result_text

# Function to get relevant content from vector store
def query_vector_store(vector_db, user_query, k=3):
    """Retrieve relevant content from PDFs and images using vector similarity search"""
    if vector_db is None:
        return "No vector store available for PDF and image data."
    
    # Search for relevant documents
    docs = vector_db.similarity_search(user_query, k=k)
    
    # Format the results
    result_text = "PDF and Image Content:\n\n"
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown source")
        doc_type = doc.metadata.get("type", "Unknown type")
        
        result_text += f"Document {i+1} ({doc_type} - {source}):\n"
        result_text += doc.page_content + "\n\n"
    
    return result_text

# Step 2: Setup prompt template and chain

CUSTOM_PROMPT_TEMPLATE = """
Use the following information sources and chat history to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answers concise and relevant to the data provided.

Database Information: 
{db_results}

PDF and Image Information:
{pdf_image_results}

Chat History: {chat_history}
Question: {question}

Start the answer directly. No small talk please. Indicate which source(s) you used to answer the question.
"""

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question"  # Specify which input key contains the user's message
)

# Create prompt template
prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["db_results", "pdf_image_results", "chat_history", "question"]
)

# Create LLM chain
llm_chain = LLMChain(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Load vector store for PDFs and images
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    vector_db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully for PDFs and images.")
except Exception as e:
    print(f"Warning: Could not load vector store: {e}")
    vector_db = None

# Interactive chat loop
print("Chat with your multimodal database! Type 'exit' to end the conversation.")
print("(Querying database and vector store with PDFs and images)")
while True:
    user_query = input("\nYou: ")
    if user_query.lower() == 'exit':
        print("Goodbye!")
        break
    
    # Get database results for every query
    db_results = query_database(user_query)
    
    # Get relevant content from PDFs and images
    pdf_image_results = query_vector_store(vector_db, user_query)
    
    # Run the chain
    response = llm_chain.invoke({
        "db_results": db_results,
        "pdf_image_results": pdf_image_results,
        "question": user_query
    })
    
    print("\nAssistant:", response["text"])