# import streamlit as st
# from openai import OpenAI

# # Show title and description.
# st.title("💬 Chatbot")
# st.write(
#     "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
#     "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
# )

# # Ask user for their OpenAI API key via `st.text_input`.
# # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# openai_api_key = st.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# else:

#     # Create an OpenAI client.
#     client = OpenAI(api_key=openai_api_key)

#     # Create a session state variable to store the chat messages. This ensures that the
#     # messages persist across reruns.
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display the existing chat messages via `st.chat_message`.
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Create a chat input field to allow the user to enter a message. This will display
#     # automatically at the bottom of the page.
#     if prompt := st.chat_input("What is up?"):

#         # Store and display the current prompt.
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Generate a response using the OpenAI API.
#         stream = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         )

#         # Stream the response to the chat using `st.write_stream`, then store it in 
#         # session state.
#         with st.chat_message("assistant"):
#             response = st.write_stream(stream)
#         st.session_state.messages.append({"role": "assistant", "content": response})
# import streamlit as st
# import sqlite3
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.embeddings import SpacyEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
# from dotenv import load_dotenv
# from langchain_anthropic import ChatAnthropic
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# import os

# # Load environment variables
# load_dotenv()

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# def db_read(db_file):
#     conn = sqlite3.connect(db_file)
#     cursor = conn.cursor()
    
#     # Adjust this query based on your table and columns
#     cursor.execute("SELECT content FROM your_table_name")
#     rows = cursor.fetchall()
    
#     text = ""
#     for row in rows:
#         text += row[0]
    
#     conn.close()
#     return text

# def get_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def vector_store(text_chunks):
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_db")

# def get_conversational_chain(tools, ques):
#     anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
#     if not anthropic_api_key:
#         raise ValueError("Anthropic API key is not set in the environment variables.")
    
#     llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key=anthropic_api_key, verbose=True)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#                 provided context just say, "answer is not available in the context", don't provide the wrong answer""",
#             ),
#             ("placeholder", "{chat_history}"),
#             ("human", "{input}"),
#             ("placeholder", "{agent_scratchpad}"),
#         ]
#     )
#     tool = [tools]
#     agent = create_tool_calling_agent(llm, tool, prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
#     response = agent_executor.invoke({"input": ques})
#     print(response)
#     st.write("Reply: ", response['output'])

# def user_input(user_question):
#     new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
#     retriever = new_db.as_retriever()
#     retrieval_chain = create_retriever_tool(retriever, "db_extractor", "This tool is to give answer to queries from the SQLite database")
#     get_conversational_chain(retrieval_chain, user_question)

# def main():
#     st.set_page_config(page_title="Chat with SQLite Database")
#     st.header("Chat with your SQLite Database")

#     user_question = st.text_input("Ask a Question from the SQLite Database")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         db_file = st.file_uploader("Upload your SQLite Database File (.sqlite3)", type=["sqlite3"])
#         if st.button("Submit & Process"):
#             if db_file:
#                 with st.spinner("Processing..."):
#                     raw_text = db_read(db_file)
#                     text_chunks = get_chunks(raw_text)
#                     vector_store(text_chunks)
#                     st.success("Done")

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import sqlite3
# import json
# import tempfile
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.embeddings import SpacyEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
# from dotenv import load_dotenv
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from together import Together
# import os

# # Load environment variables
# load_dotenv()

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# def db_read(db_file_path):
#     conn = sqlite3.connect(db_file_path)
#     cursor = conn.cursor()
    
#     # Adjust this query based on your table and columns
#     cursor.execute("SELECT content FROM documents")
#     rows = cursor.fetchall()
    
#     data = []
#     for row in rows:
#         data.append({"content": row[0]})
    
#     conn.close()
#     return json.dumps(data)

# def get_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def vector_store(text_chunks):
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_db")

# def get_conversational_chain(tools, ques):
#     together_api_key = os.getenv("TOGETHER_API_KEY")
#     if not together_api_key:
#         raise ValueError("Together API key is not set in the environment variables.")
    
#     client = Together(api_key=together_api_key)
    
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, 'answer is not available in the context'. Do not provide the wrong answer."},
#         {"role": "user", "content": ques}
#     ]
    
#     try:
#         response = client.chat.completions.create(
#             model="meta-llama/Llama-3-8b-chat-hf",
#             messages=messages,
#             max_tokens=512,
#             temperature=0.7,
#             top_p=0.7,
#             top_k=50,
#             repetition_penalty=1,
#             stop=["", ""],
#             stream=False
#         )
        
#         print(response.choices[0].message.content)
#         st.write("Reply: ", response.choices[0].message.content)
#     except Exception as e:
#         st.error(f"API Error: {str(e)}")
#         print(f"API Error: {str(e)}")

# def user_input(user_question):
#     new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
#     retriever = new_db.as_retriever()
#     retrieval_chain = create_retriever_tool(retriever, "db_extractor", "This tool is to give answer to queries from the SQLite database")
#     get_conversational_chain(retrieval_chain, user_question)

# def main():
#     st.set_page_config(page_title="Chat with SQLite Database")
#     st.header("Chat with your SQLite Database")

#     user_question = st.text_input("Ask a Question from the SQLite Database")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         db_file = st.file_uploader("Upload your SQLite Database File (.sqlite3)", type=["sqlite3"])
#         if st.button("Submit & Process"):
#             if db_file:
#                 with st.spinner("Processing..."):
#                     # Save the uploaded file to a temporary location
#                     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#                         temp_file.write(db_file.read())
#                         temp_file_path = temp_file.name

#                     raw_text = db_read(temp_file_path)
#                     json_data = json.loads(raw_text)
#                     text = " ".join([entry["content"] for entry in json_data])
#                     text_chunks = get_chunks(text)
#                     vector_store(text_chunks)
#                     st.success("Done")

# if __name__ == "__main__":
#     main()

# gpt4-0

# import streamlit as st
# import sqlite3
# import json
# import tempfile
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.embeddings import SpacyEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Configure Google Generative AI
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# def db_read(db_file_path):
#     conn = sqlite3.connect(db_file_path)
#     cursor = conn.cursor()
    
#     # Adjust this query based on your table and columns
#     cursor.execute("SELECT content FROM documents")
#     rows = cursor.fetchall()
    
#     data = []
#     for row in rows:
#         data.append({"content": row[0]})
    
#     conn.close()
#     return json.dumps(data)

# def get_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def vector_store(text_chunks):
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_db")

# def retrieve_context(retriever, ques):
#     results = retriever.invoke(ques)
#     context = " ".join([result.page_content for result in results])
#     return context

# def get_conversational_chain(context, ques):
#     prompt = f"This is the context from the SQLite database: {context}\n\nQuestion: {ques}\n\nAnswer:"

#     try:
#         response = genai.generate_text(prompt=prompt)
#         answer = response.candidates[0].text
        
#         print(answer)
#         st.write("Reply: ", answer)
#     except Exception as e:
#         st.error(f"API Error: {str(e)}")
#         print(f"API Error: {str(e)}")

# def user_input(user_question):
#     new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
#     retriever = new_db.as_retriever()
#     context = retrieve_context(retriever, user_question)
#     get_conversational_chain(context, user_question)

# def main():
#     st.set_page_config(page_title="Chat with SQLite Database")
#     st.header("Chat with your SQLite Database")

#     user_question = st.text_input("Ask a Question from the SQLite Database")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         db_file = st.file_uploader("Upload your SQLite Database File (.sqlite3)", type=["sqlite3"])
#         if st.button("Submit & Process"):
#             if db_file:
#                 with st.spinner("Processing..."):
#                     # Save the uploaded file to a temporary location
#                     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#                         temp_file.write(db_file.read())
#                         temp_file_path = temp_file.name

#                     raw_text = db_read(temp_file_path)
#                     json_data = json.loads(raw_text)
#                     text = " ".join([entry["content"] for entry in json_data])
#                     text_chunks = get_chunks(text)
#                     vector_store(text_chunks)
#                     st.success("Done")

# if __name__ == "__main__":
#     main()

import os
import json
import tempfile
import sqlite3
import google.generativeai as genai
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def db_read(db_file_path):
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # Adjust this query based on your table and columns
    cursor.execute("SELECT content FROM documents")
    rows = cursor.fetchall()
    
    data = []
    for row in rows:
        data.append({"content": row[0]})
    
    conn.close()
    return json.dumps(data)

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def retrieve_context(retriever, ques):
    try:
        # Use 'invoke' method for the retriever
        results = retriever.invoke(input=ques)
        context = " ".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        st.error(f"Retrieval Error: {str(e)}")
        print(f"Retrieval Error: {str(e)}")
        return ""

def get_conversational_chain(context, ques):
    prompt = f"This is the context from the SQLite database: {context}\n\nQuestion: {ques}\n\nAnswer:"

    try:
        # Generative API call
        response = genai.generate_text(prompt=prompt)

        # Print the response for debugging
        print("API Response:", response)

        # Check if the response has the expected structure
        if isinstance(response, dict):
            if 'result' in response:
                answer = response['result']
            elif 'text' in response:
                answer = response['text']
            else:
                answer = "Unexpected response format: 'result' or 'text' key missing."
        elif hasattr(response, 'result'):
            answer = response.result
        elif hasattr(response, 'text'):
            answer = response.text
        else:
            answer = "Unexpected response format: Response does not have 'result' or 'text' attribute."

        print("Generated Answer:", answer)
        st.write("Reply: ", answer)
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        print(f"API Error: {str(e)}")

def user_input(user_question):
    try:
        new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()
        context = retrieve_context(retriever, user_question)
        get_conversational_chain(context, user_question)
    except Exception as e:
        st.error(f"User Input Error: {str(e)}")
        print(f"User Input Error: {str(e)}")

def main():
    st.set_page_config(page_title="Chat with SQLite Database")
    st.header("Chat with your SQLite Database")

    user_question = st.text_input("Ask a Question from the SQLite Database")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        db_file = st.file_uploader("Upload your SQLite Database File (.sqlite3)", type=["sqlite3"])
        if st.button("Submit & Process"):
            if db_file:
                with st.spinner("Processing..."):
                    # Save the uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(db_file.read())
                        temp_file_path = temp_file.name

                    raw_text = db_read(temp_file_path)
                    json_data = json.loads(raw_text)
                    text = " ".join([entry["content"] for entry in json_data])
                    text_chunks = get_chunks(text)
                    vector_store(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    main()
