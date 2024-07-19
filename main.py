import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;

    Your turn:

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    os.environ["NVIDIA_API_KEY"] = "#"
    llm = ChatNVIDIA(base_url="https://integrate.api.nvidia.com/v1", model="mistralai/codestral-22b-instruct-v0.1")

    def get_schema(_):
        return db.get_table_info()

    return (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
    )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""

    prompt = ChatPromptTemplate.from_template(template)

    os.environ["NVIDIA_API_KEY"] = "#"
    llm = ChatNVIDIA(base_url="https://integrate.api.nvidia.com/v1", model="mistralai/codestral-22b-instruct-v0.1")

    chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]),
            )
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })


def main():
    load_dotenv()

    # Initialize database connection
    db = init_database(
        user="majna",
        password="12345678",
        host="localhost",
        port="5432",
        database="majna"
    )

    chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

    print("Chat with PostgreSQL")
    print("Type 'exit' to end the conversation.")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break

        chat_history.append(HumanMessage(content=user_query))

        response = get_response(user_query, db, chat_history)
        print("AI:", response)

        chat_history.append(AIMessage(content=response))


if __name__ == "__main__":
    main()
