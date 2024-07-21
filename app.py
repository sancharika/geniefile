# app.py
import os
import streamlit as st
from components import functions
from components.docLoader import docLoader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_core.documents import Document
from langchain_groq import ChatGroq


st.set_page_config(page_title="geniefile", page_icon='🗃️', layout='wide')




# Load environment variables
load_dotenv()

# Initialize LLMModel
class LLMModel(object):
    def __init__(self):
        pass

    @staticmethod
    def model():
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return ChatGoogleGenerativeAI(model= "gemini-1.5-flash-latest",
                 temperature=0.3, top_p=0.85)

    @staticmethod
    def embeddings():
         return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    @staticmethod
    def graphModel():
        return ChatGroq(
    temperature=0,
    model="gemma2-9b-It",
)
        

# Initialize LLMModel instance
llm_model = LLMModel()
func = functions.Functions()
# Set Streamlit page configuration

# Main title
st.title('🪄 GenieFile 🪄')


graph_model = llm_model.graphModel()
embeddings = llm_model.embeddings()

demo = st.toggle("Try Example")

input_demo = ""
if demo:
    #read document file paper
    # Get the current working directory
    current_dir = os.getcwd()

    # Construct the path to the file
    uploaded_file = os.path.join(current_dir, 'file', 'Paper35.pdf')
    docs_loader = docLoader(uploaded_file)
    docs = docs_loader.load_pdf()
    input_demo = "Tell me about this doc"
    st.caption("This doc already exist in Database you dont have to save it again")

else:
    # Load document
    uploaded_file = st.file_uploader("Choose a document file", type=["pdf", "txt", "docx"])
    docs_loader = docLoader(uploaded_file)
    docs = docs_loader.load()
    st.caption("Please reset existing database and save the doc in database")

    
st.session_state['doc_text'] = docs


# Sidebar options
with st.sidebar:    
    # Load model
    with st.spinner("Loading Model..."):
        llm = llm_model.model()
    
    add, delete = st.columns(2)
    with add:
        add_data = st.button(':green[Save Data]')
    with delete:
        reset = st.button("Reset Database", type="primary")
    
    if docs:
        extracted= st.text_area("Extracted Data", value=st.session_state['doc_text'])
        
    if st.session_state['doc_text']:
        docs_loader.display_doc()


if add_data:
        with st.spinner("Saving Data in Database. Please wait it may take more than couple minutes.."):
            func.add_data(docs, embeddings, graph_model)
        st.success('Data Saved in Database', icon="✅")

if reset:
        with st.spinner("Deleting Data from Database. Please wait it may take more than couple minutes.."):
            func.delete_db()
        st.warning('Database reset Successful', icon="🗑️")
            
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.demo_processed = False

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add the demo message to chat history and display it
if input_demo and not st.session_state.demo_processed:
    st.session_state.messages.append({"role": "user", "content": input_demo})
    with st.chat_message("user"):
        st.markdown(input_demo)

    # Generate and display a response for the demo input
    with st.spinner("Loading Answer..."):

        response = func.retrieve_answers(input_demo, llm, docs, embeddings)
        st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write_stream(func.response_generator(response))

    # Mark demo message as processed
    st.session_state.demo_processed = True


# React to user input
if prompt := st.chat_input("Shoot your question?"):
    print(st.session_state.demo_processed)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display a response for the demo input
    with st.spinner("Loading Answer..."):

        response = func.retrieve_answers(prompt, llm, docs, embeddings)
        st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write_stream(func.response_generator(response))

    