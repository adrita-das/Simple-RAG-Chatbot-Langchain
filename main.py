import streamlit as st
from pypdf import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_document(user_input):
    # user_input = st.chat_input('Ask a question' , accept_file='multiple' , file_type='pdf')
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
                        
    
    return text 

# get_document()    
               


def get_chunk(chunk):
    
    text_spliter = RecursiveCharacterTextSplitter(
        # separators= 
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len

    )
    
    chunks = text_spliter.split_text(chunk)
    
    return chunks


def main():
    st.set_page_config(page_title='Chat with URL & PDF', page_icon=':books:')
    
    user_input = st.chat_input('Ask a question', accept_file='multiple', file_type='pdf')
    
    with st.sidebar:
        raw_text = get_document(user_input)
        
        if raw_text:
            st.write(raw_text)
            chunk_text = get_chunk(raw_text)
            
            if chunk_text:
                st.write(chunk_text)
    
    
if __name__ == '__main__':
    main()  
    
    
    

