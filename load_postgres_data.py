from langchain.docstore.document import Document
import psycopg2

def load_postgres_data(host, port, dbname, user, password, query):
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    
    # Get column names
    column_names = [desc[0] for desc in cur.description]
    
    cur.close()
    conn.close()

    # Create documents with proper metadata
    documents = []
    for row in rows:
        # Create a dictionary of column names and values
        row_dict = dict(zip(column_names, row))
        
        # Create a readable string representation of the data
        content = ", ".join([f"{col}: {val}" for col, val in row_dict.items()])
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "source": f"database/{dbname}",
                "type": "database_record",
                "table": query.split("FROM")[1].split()[0].strip() if "FROM" in query else "unknown_table"
            }
        )
        documents.append(doc)
        
    return documents
