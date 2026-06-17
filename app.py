import streamlit as st
import os
import cohere
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
#from qdrant_client import Distance,

load_dotenv()

def get_text(pdf_files, url_input):
    
    text = ''
    
    #Handle pdf
    if pdf_files:        
        for pdf in pdf_files:
            read_pdf = PdfReader(pdf)
            
            for page in read_pdf.pages:
                text += page.extract_text()
    
    # Handle URL
    if url_input:
        read_url = WebBaseLoader(web_paths=[url_input])
        
        url = read_url.load()
        
        for doc in url:
            text += doc.page_content
    
    return text   

def get_chunk(chunk):
    text_spilter = RecursiveCharacterTextSplitter(
        # separators=['\n'],
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    
    chunks = text_spilter.split_text(chunk)
    return chunks

def get_vector_store(chunk_text):
    embeddings = CohereEmbeddings(
        model='embed-english-v3.0',
        cohere_api_key=os.getenv('COHERE_API_KEY')
    )

    vectorStore = QdrantVectorStore.from_texts(
        texts=chunk_text,
        embedding=embeddings,
        path='qdrant_storage',
        collection_name='rag_collection',
        force_recreate=True
    )

    return vectorStore
    
            
def main():
    # load_dotenv()
    st.set_page_config(page_title='Chat with Documents', page_icon=':books:')
    
    st.header('Chat with Docs:books:')
    user_que=st.text_input('Ask a question')
    
    with st.sidebar:
        st.subheader('Your Document')
        pdf_files=st.file_uploader('Upload your documents here.You can upload Multiple documents' , accept_multiple_files= True)
        
        url_input=st.text_input('Your Web URL', placeholder='https://example.com')
        
        if st.button('Process'):
            with st.spinner('Processing'):
                
                # get the text
                raw_text= get_text(pdf_files, url_input)
                # st.write(raw_text)
                
                # get the chunks 
                chunk_text = get_chunk(raw_text)
                #st.write(chunk_text)
                
                #create vectorstore
                vectorStore = get_vector_store(chunk_text)
                
                
    
if __name__ == '__main__':
    main()    

