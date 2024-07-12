# app.py
import os
import streamlit as st
from components import docLoader, functions
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq



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
st.set_page_config(page_title="geniefile", page_icon='🗃️', layout='wide')

# Main title
st.title('🪄 GenieFile 🪄')


# Load document
docs = docLoader.load_doc()

st.session_state['doc_text'] = docs

graph_model = llm_model.graphModel()
embeddings = llm_model.embeddings()
doc = st.columns(1)
if docs: 
    # with doc:
        extracted= st.text_area("Extracted Data", value=st.session_state['doc_text'])


        
input_text = st.text_input("Enter Your Question")


# Sidebar options
with st.sidebar:    
    # Load model
    with st.spinner("Loading Model..."):
        llm = llm_model.model()
    
    add, delete = st.columns(2)
    with add:
        add_data = st.button(':green[Save Data]')
    with delete:
        reset = st.button("Reset", type="primary")
    
    if add_data:
        func.add_data(docs, embeddings, graph_model)
        st.success('Data Saved in Database', icon="✅")

    if reset:
         
        func.delete_db()
        st.warning('Database reset Successful', icon="🗑️")
        
        
    
    submit=st.button('ask')


if submit or input_text:
            answer, graph_answer = func.retrieve_answers(input_text, llm, docs, embeddings)
            print(answer)
            st.write("Answer generated from Knowledge Graph: "+graph_answer)
            st.write("Answer generated from FAISS: "+answer)


