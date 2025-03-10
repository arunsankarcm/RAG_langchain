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
    cur.close()
    conn.close()

    documents = [Document(page_content=str(row)) for row in rows]
    return documents
