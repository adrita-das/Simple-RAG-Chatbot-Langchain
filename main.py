import streamlit as st
import os 
from pypdf import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


load_dotenv()

def get_document(user_input):
    
    document = []

    # text = ''
    
    # read pdf
    if user_input and user_input.files:
        pdf = user_input.files
        for pdf_files in pdf:
            read_pdf = PdfReader(pdf_files)
            
            for page_num,page in enumerate(read_pdf.pages, start=1):
                page_text = page.extract_text()
                
                document.append(Document(page_content = page_text , metadata= {"page" : page_num }
                    
                 ))  
                
     # read from url 
    
    if user_input and user_input.text:
        read_url = WebBaseLoader(user_input.text)
        url = read_url.load()
        document.extend(url)
                        
    
    return document

def get_chunk(chunk):
    
    text_spliter = RecursiveCharacterTextSplitter(
        
        chunk_size = 1000,
        chunk_overlap = 200,

    )
    
    chunks = text_spliter.split_documents(chunk)
    
    return chunks

def get_embeddigs(chunk_text):
    embeddings = CohereEmbeddings(
        model='embed-english-v3.0',
        cohere_api_key=os.getenv('COHERE_API_KEY')
    )
    
    vector_db = QdrantVectorStore.from_documents(
        documents = chunk_text,
        embedding= embeddings,
        path= 'qdrant_storage',
        collection_name='rag_collection',
        
    )
    
    return vector_db
    

def main():
    st.set_page_config(page_title='Chat with URL & PDF', page_icon=':books:')
    
    user_input = st.chat_input('Ask a question', accept_file='multiple', file_type='pdf')
    
    raw_text = get_document(user_input)
    
    # with st.sidebar:
    #     raw_text = get_document(user_input)
        
    #     if raw_text:
    #         # st.write(raw_text)
    #         chunk_text = get_chunk(raw_text)
            
    #         if chunk_text:
    #              st.write(chunk_text)
    
    if raw_text:
        chunk_text = get_chunk(raw_text)
        # return chunk_text
        
    vectorStore = get_embeddigs(chunk_text)    
        
    
           
if __name__ == '__main__':
    main()  
    
    
    

