import streamlit as st
import os
import cohere
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

load_dotenv()

def get_text(pdf_files, url_input):
    text = ''
    
    # Handle pdf
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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
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
        path= 'qdrant_storage',
        collection_name='rag_collection',
        force_recreate=True
    )

    return vectorStore

def get_user_que(user_que, vector_store):
    llm = ChatCohere(
        cohere_api_key = os.getenv('COHERE_API_KEY'),
        model = 'command-r-08-2024', 
        temperature = 0.7,
        max_tokens = 512
    )
    
    # search vector database
    search_database = vector_store.similarity_search(user_que, k=4)
    
    # combine matching into chunk
    context = ''
    for doc in search_database:
        context += doc.page_content + "\n\n"
        
    final_input = f"Context:\n{context}\n\nQuestion: {user_que}\n\nAnswer the question using the context above."
    
    # 5. Get answer from Cohere
    response = llm.invoke(final_input)
    return response.content     
            
def main():
    st.set_page_config(page_title='Chat with Documents', page_icon=':books:')
    st.header('Chat with Docs :books:')
    
    
    with st.sidebar:
        st.subheader('Your Document')
        pdf_files = st.file_uploader('Upload your documents here. You can upload multiple documents', accept_multiple_files=True)
        url_input = st.text_input('Your Web URL', placeholder='https://example.com')
        
        if st.button('Process'):
            with st.spinner('Processing...'):
                # 1. Get the text
                raw_text = get_text(pdf_files, url_input)
                
                # 2. Get the chunks 
                chunk_text = get_chunk(raw_text)
                
                # 3. Create vectorstore in RAM
                vectorStore = get_vector_store(chunk_text)
                
                # 4. Save to session state so it remembers the data when typing a question
                st.session_state.vector_store = vectorStore
                
                st.success("Documents processed successfully!")
                
    user_que = st.text_input('Ask a question')   
    
    if user_que:
        
        if 'vector_store' in st.session_state:
            with st.spinner('Thinking...'):
                answer = get_user_que(user_que, st.session_state.vector_store)
                st.write(answer)
                
        else:
            st.info("Please upload and process documents in the sidebar first.")
                    
if __name__ == '__main__':
    main()