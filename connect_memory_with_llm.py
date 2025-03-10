import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from create_memory_for_llm import create_memory_for_llm

# Step 1: Setup Local Llama Model
MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"  # Local path to the downloaded model

def load_llm():
    # Initialize Llama model with ctransformers
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load or create Database
DB_FAISS_PATH = "vectorstore/db_faiss"

try:
    # Try to load existing database
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except:
    # If loading fails, create new database
    print("Creating new vector database...")
    db = create_memory_for_llm()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
if __name__ == "__main__":
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})
    print("RESULT: ", response["result"])
    print("SOURCE DOCUMENTS: ", response["source_documents"])