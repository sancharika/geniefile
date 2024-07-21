#fucntions.py
import os
import time
import pyperclip
import streamlit as st
import speech_recognition as sr
from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders.telegram import text_to_docs
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector




class Functions():
    def __init__(self):
        pass
    

    @staticmethod
    def get_gemini_response(llm, input_text, doc, template, info=''):
        formated_prompt = template.format(doc=doc, input_text=input_text, info=info)
        response = llm.invoke(formated_prompt)
        return response.content
        # return formated_prompt

    @staticmethod
    def copy_text(answer, copy_button=False):
        pyperclip.copy(answer)
        if copy_button:
            st.toast("Text copied to clipboard!", icon="📋")

    @staticmethod
    def record_audio():
        r = sr.Recognizer()
        with st.spinner("Recording..."):
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                with st.spinner("Say Something..."):
                    audio = r.listen(source, timeout=5)
            with st.spinner("Processing..."):
                try:
                    text = r.recognize_google(audio)
                    st.session_state['input_text'] = text
                    return text
                except sr.UnknownValueError:
                    st.write("Sorry, I could not understand what you said. Please try again or write in text box.")
                    return ""
                except sr.RequestError as e:
                    st.write(f"Could not request results; {e}")
                    return ""

    @staticmethod
    def input_state(input_text):
        if isinstance(input_text, str):
            st.session_state['input_text'] = input_text

    @staticmethod
    def delete_db():
        # delte vector db
        file_path = 'saved_embeddings'
        # Check if the file exists
        if os.path.exists(file_path+'\index.faiss'):
            # Delete the file
            os.remove(file_path+'\index.faiss')
        if os.path.exists(file_path+'\index.pkl'):
        # Delete the file
            os.remove(file_path+'\index.pkl')
        st.toast("File Deleted from Vector DataBase!", icon="⚠️")

        # delete the graph
        uri = os.getenv("NEO4J_URI") 
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        st.toast("File deleted From Knowledge Graph!", icon="⚠️")
    
    @staticmethod
    def add_data(data, gemini_embeddings, graph_model, file_path="saved_embeddings"):
        def chunk_data(docs,chunk_size=10000,chunk_overlap=5000):
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            doc=text_splitter.split_documents(docs)
            return doc
        
        def add_graph(data, model, batch_size=64):
            llm_transformer = LLMGraphTransformer(llm=model)
            print(f'---------------------------total docs {len(data)}-----------------------------------')
            for i in range(0, len(data), batch_size):
                print(f'---------------------------Iteraion {i} started-----------------------------------')
                documents = data[i: i+batch_size]
                graph_documents = llm_transformer.convert_to_graph_documents(documents)
                print(f"Nodes:{graph_documents[0].nodes}")
                print(f"Relationships:{graph_documents[0].relationships}")
                graph = Neo4jGraph()
                graph.add_graph_documents(
                graph_documents, 
                baseEntityLabel=True, 
                include_source=True
                )
                print(f'---------------------------Iteraion {i} completed-----------------------------------')

            st.toast("File uploaded successfully to Knowledge Graph!", icon="✅")

        docs = chunk_data(text_to_docs(data))
        db = FAISS.from_documents(docs, gemini_embeddings)
        # saving the document in the vector store
        db.save_local(file_path)
        st.toast("File uploaded To Vector DataBase!", icon="✅")
        add_graph(docs, graph_model)    
    
    @staticmethod
    def retrieve_answers(query, llm, data, gemini_embeddings, file_path="saved_embeddings"):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in text_to_docs(data))
        
        tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        #generate answer from vector db FAISS
        # reloading the documents from the vector store
        new_db = FAISS.load_local(folder_path=file_path, embeddings =gemini_embeddings,
                                    allow_dangerous_deserialization = True)

        # Prompt template to query Gemini
        llm_prompt_template = """You are an assistant for question-answering tasks with advanced analytical and reasoning capabilities
        Use the following context to answer the question.
        If you don't know the answer, just say that you don't know.
        Keep the answer concise.\n
        Question: {question} \nContext: {context} \nAnswer:"""

        llm_prompt = PromptTemplate.from_template(llm_prompt_template)
        
        # creating a retriver object 
        retriever = new_db.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever= retriever)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | llm_prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)

        vector_index = Neo4jVector.from_existing_graph(
            gemini_embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        
        # generate answr from knowledge graph

        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, chain_type="stuff", retriever=vector_index.as_retriever()
        )
        result = qa_chain.invoke(
            {"question": query},
            return_only_outputs=True,
        )
        #unifying answers
        
        faiss_answer = response
        graph_answer = result['answer'].split('FINAL ANSWER:')[-1]
        # Combine both answers
        combined_answer = faiss_answer + " " + graph_answer
        
        combined_tokens = tokenizer(combined_answer, return_tensors="pt").input_ids.shape[1]
        faiss_tokens = tokenizer(faiss_answer, return_tensors="pt").input_ids.shape[1]
        graph_tokens = tokenizer(graph_answer, return_tensors="pt").input_ids.shape[1]
        
        # Use a summarizer to merge the content and remove redundancy
        summarizer = pipeline("summarization")
        summary = summarizer(combined_answer, max_length=combined_tokens, min_length=max(faiss_tokens, graph_tokens), do_sample=False)
        
        final_answer = summary[0]['summary_text']
        
        return final_answer
    
    @staticmethod
    # Streamed response emulator
    def response_generator(response):
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

