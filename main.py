import streamlit as st
import os 
from pypdf import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings,ChatCohere
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient



load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# read from pdf 

def get_document(target_pdf):
    
    document = []

    if target_pdf:
        for pdf in target_pdf:
            read_pdf = PdfReader(pdf)
            
            for page_num,page in enumerate(read_pdf.pages, start=1):
                page_text = page.extract_text()
                
                document.append(Document(page_content = page_text , metadata= {"page" : page_num }
                    
                 ))  
            
    return document

# read from url

def get_url(target_url):
    
    url = []
    
    if target_url:
        try:
            read_url = WebBaseLoader(web_paths=[target_url])
            url.extend(read_url.load())
            
        except Exception:
            pass
             
    return url    

# create chunk
    
def get_chunk(chunk):
    
    text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
    
    chunks = text_spliter.split_documents(chunk)
    
    return chunks

# embeddings & vector database 

def get_embeddigs(chunk_text):
    embeddings = CohereEmbeddings(
        model='embed-english-v3.0',
         cohere_api_key=os.getenv('COHERE_API_KEY')
    )
    
        
    vector_db = QdrantVectorStore.from_documents(
        documents = chunk_text,
        embedding= embeddings,
        location=':memory:',
        #path= 'qdrant_storage',
        collection_name='rag_collection',
        force_recreate=True
            
    )
    
    return vector_db


def user_query(vector_store):
    llm = ChatCohere(
        
        cohere_api_key = os.getenv('COHERE_API_KEY'),
        model = 'command-r-08-2024', 
        temperature = 0.7,
        max_tokens = 512
        
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vector_store.as_retriever()
    )
    
    return conversation_chain
      
def main():
    st.set_page_config(page_title='Chat with URL & PDF', page_icon=':books:')
    
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
        
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
        
    for role,message in st.session_state.conversation:
        with st.chat_message(role):
            st.write(message)

                        
    user_input = st.chat_input('Ask a question', accept_file='multiple', file_type='pdf' , key='url')
    
    if user_input:
        pdf_files = user_input.files      
        text_input = user_input.text
        
        raw_text = get_document(pdf_files)
        
        if raw_text:
             chunk_text = get_chunk(raw_text)
             vectorStore = get_embeddigs(chunk_text)
             st.session_state.vector_store = vectorStore
            

        raw_text_url = get_url(text_input)  
        
        if raw_text_url:
            chunk_text_url = get_chunk(raw_text_url)
            vectorStoreUrl =get_embeddigs(chunk_text_url)
            st.session_state.vector_store = vectorStoreUrl  
        
        msg = text_input if text_input else ''
        
        if pdf_files:
            for f in pdf_files:
                msg = ":material/attach_file: " + f.name + "  \n\n" + msg
                st.write(msg)
                
        
        if msg:
            st.session_state.conversation.append(('user' , msg))
            with st.chat_message('user'):
                st.markdown(msg)
                
            if 'vector_store' in st.session_state:
                chain = user_query(st.session_state.vector_store)
                
                with st.chat_message('assistant'):
                    with st.spinner('Thinking....'):
                        response = chain.invoke({'question' : text_input , 'chat_history' : []})
                        answer = response['answer']
                        st.markdown(answer)
                        
                st.session_state.conversation.append(('assistant' , answer))
            
        else:
            st.warning('Please upload a PDF or URL first')                      
           
if __name__ == '__main__':
    main()  
    
    
    

