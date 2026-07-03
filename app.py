import streamlit as st
import os
import cohere
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document


load_dotenv()
co = cohere.Client(os.getenv('COHERE_API_KEY'))

def get_documents(pdf_files, url_input):
    # text = ''
    document = []
    
    # Handle pdf
    if pdf_files:        
        for pdf in pdf_files:
            read_pdf = PdfReader(pdf)
            for page_num, page in enumerate(read_pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    document.append(Document(
                        page_content=text,
                        metadata = {"source" : pdf.name , "page": page_num}
                    ))
                
    
    # Handle URL
    if url_input:
        read_url = WebBaseLoader(web_paths=[url_input])
        document.extend(read_url.load())
        
    return document    
  

def get_chunk(document):
    text_spilter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_spilter.split_documents(document)
    

def get_vector_store(chunks):
    embeddings = CohereEmbeddings(
        model='embed-english-v3.0',
         cohere_api_key=os.getenv('COHERE_API_KEY')
    )

    vectorStore = QdrantVectorStore.from_documents(
        documents = chunks,
        embedding=embeddings,
        location=':memory:',
        #path= 'qdrant_storage',
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
    docs = vector_store.similarity_search(user_que, k = 15)
    
    re_ranked = co.rerank(
        query = user_que,
        documents= [doc.page_content for doc in docs],
        model = 'rerank-v4.0-pro',
        top_n = 4
    )
    
    best_docs = [docs[r.index] for r in re_ranked.results]
    
    # build documents for citations 
    
    document = [
        
        {"id" : f'chunk_{i}' , 
         "data" : doc.page_content
         }
        
        for i, doc in enumerate(best_docs)
    ]
    
    # generate answer with citations 
    response = co.chat (
        message = user_que,
        documents = document,
        model = 'command-r-08-2024'
        )
    
    citations_out = []
    
    if response.citations:
        for citation in response.citations:
            for doc_id in citation.document_ids:
                index = int(doc_id.split('_')[1])
                citations_out.append({
                
                    "text" : citation.text,
                    "page" : best_docs[index].metadata.get('page', 'N/A')
                    }
                )
                
    return response.text , citations_out            
                

def main():
    st.set_page_config(page_title='Chat with Documents', page_icon=':books:')
    st.header('Chat with Docs :books:')

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader('Your Document')
        pdf_files = st.file_uploader('Upload your documents here. You can upload multiple documents', accept_multiple_files=True)
        url_input = st.text_input('Your Web URL', placeholder='https://example.com')

        if st.button('Process'):
            with st.spinner('Processing...'):
                raw_docs = get_documents(pdf_files, url_input)
                chunk_text = get_chunk(raw_docs)
                vectorStore = get_vector_store(chunk_text)
                st.session_state.vector_store = vectorStore
                # st.session_state.messages = []  
                st.success("Documents processed successfully!")

    # Replay the full conversation so far
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("citations"):
                with st.expander("Sources"):
                    for c in msg["citations"]:
                        st.caption(f'"{c["text"]}" - page {c["page"]}')

    # Chat input — clears automatically after each question
    user_que = st.chat_input("Ask a question about your document")

    if user_que:
        if st.session_state.vector_store is None:
            st.error("Please upload and process a document first.")
        else:
            with st.chat_message("user"):
                st.write(user_que)
            st.session_state.messages.append({"role": "user", "content": user_que})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, citations = get_user_que(user_que, st.session_state.vector_store)
                    st.write(answer)
                    if citations:
                        with st.expander("Sources"):
                            for c in citations:
                                st.caption(f'"{c["text"]}" - page {c["page"]}')
            st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})


if __name__ == '__main__':
    main()