from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

model=ChatGroq(model_name='llama-3.1-8b-instant')
prompt=ChatPromptTemplate.from_template("""Answer from the following context only.
Please provide the most accurate results based on the context.
{context}
Question: {input} """)

st.session_state.embeddings=OllamaEmbeddings(model='nomic-embed-text:latest')
st.session_state.loader=PyPDFDirectoryLoader('data')
st.session_state.docs=st.session_state.loader.load()
st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
st.session_state.splitted=st.session_state.splitter.split_documents(st.session_state.docs)
st.session_state.vectors=FAISS.from_documents(st.session_state.splitted,embedding=st.session_state.embeddings)
st.write("Vector Database is ready!")

st.title("Sunday School Class XI")
user_prompt=st.text_input("Enter your query from the uploaded document")

if st.button("Answer"):
    if user_prompt:
        document_context=create_stuff_documents_chain(model,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieved_data=create_retrieval_chain(retriever,document_context)
        response=retrieved_data.invoke({'input':user_prompt})
        st.write("""ANSWER""")
        
        st.write(response['answer'])
        
        with st.expander("Context from the documents:"):
            if "context" in response:
                for i,doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write('..............................')
        