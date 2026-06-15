import streamlit as st 
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from pypdf import PdfReader


def get_text(pdf_files, url_input):
    text = ''
    
    #Handle pdf
    
    if pdf_files:        
        for pdf in pdf_files:
            read_pdf = PdfReader(pdf)
            
            for page in read_pdf.pages:
                text += page.extract_text()
    
    if url_input:
        read_url = WebBaseLoader(url_input)
        
        url = read_url.load()
        
        for doc in url:
            text += doc.page_content
    
    return text         
            
                
def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with URL & PDF', page_icon=':books:')
    
    st.header('Chat with URL & PDF :books:')
    
    user_que=st.text_input('Ask a question')
    
    with st.sidebar:
        st.subheader('Your Document')
        pdf_files=st.file_uploader('Upload your PDF here' , accept_multiple_files= True,  type= ['pdf'])
        
        url_input=st.text_input('Your Web URL', placeholder='https://example.com')
        
        if st.button('Process'):
            with st.spinner('Processing'):
                
                # get the text
                raw_text= get_text(pdf_files, url_input)
                st.write(raw_text)
                
                
    
if __name__ == '__main__':
    main()    

