import os
from langchain.llms.base import LLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from google import genai
from typing import Optional, List
from pydantic import Field
from load_postgres_data import load_postgres_data

GEMINI_API_KEY = "AIzaSyAbDKUXZyvMFqtcTiso82wT8qnr6YAoCBw"

class GeminiFlashLLM(LLM):
    model: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.1)
    max_length: int = Field(default=512)
    client: Optional[any] = None

    @property
    def _llm_type(self) -> str:
        return "gemini_flash"

    def __init__(self, **data):
        super().__init__(**data)
        self.client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1alpha"})

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_length  
            }
        )
        return response.text

# Improved prompt template
CUSTOM_PROMPT_TEMPLATE = """
You have structured data about employees provided below. Read through it carefully and answer explicitly and precisely based ONLY on this data.
If asked comparative questions, carefully compare all relevant entries explicitly before answering.
If you don't find the answer explicitly within the provided data, reply "I don't know."

Employee Data:
{context}

Question:
{question}

Explicit Answer:
"""


# Load and format PostgreSQL data explicitly
HOST, PORT, DBNAME, USER, PASSWORD = "127.0.0.1", 5432, "travel_rag", "postgres", "experion@123"
SQL_QUERY = "SELECT name, department, email, salary, join_date FROM employees;"

db_documents = load_postgres_data(HOST, PORT, DBNAME, USER, PASSWORD, SQL_QUERY)

# Helper function to format context
def format_documents_for_context(docs):
    return "\n".join(doc.page_content for doc in docs)

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer"
)

# Set up the LLM chain
qa_chain = LLMChain(
    llm=GeminiFlashLLM(),
    prompt=PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    ),
    memory=memory,
    output_key="answer"
)

# Interactive loop
while True:
    user_query = input("Write Query Here (type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break

    context = format_documents_for_context(db_documents)
    response = qa_chain.invoke({
        'context': context,
        'question': user_query
    })
    print("\nRESULT:", response["answer"])

    print("\n" + "-"*50 + "\n")