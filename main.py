import streamlit as st
from pypdf import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_document():
    user_input = st.chat_input('Ask a question' , accept_file='multiple' , file_type='pdf')
    text = ''
    
    # read pdf
    if user_input and user_input.files:
        pdf = user_input.files
        for pdf_files in pdf:
            read_pdf = PdfReader(pdf_files)
            
            for page in read_pdf.pages:
                text += page.extract_text()
                
                
     # read from url 
    
    if user_input and user_input.text:
        read_url = WebBaseLoader(user_input.text)
        
        url = read_url.load()
        
        for doc in url:
            text +=doc.page_content
                        
    with st.sidebar:
      
        st.write(text)
    
    # return text 

get_document()    
               


def get_chunk(chunk):
    
    text_spliter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        chunk_size = 1000,
        chunk_overlap = 200
    )
    
    chunks = text_spliter.split_text(chunk)
    
    return chunks
    
    

