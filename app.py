import streamlit as st 

def main():
    st.set_page_config(page_title='Chat with URL & PDF', page_icon=':books:')
    
    st.header('Chat with URL & PDF :books:')
    
    st.text_input('Ask a question')
    
    with st.sidebar:
        st.subheader('Your Document')
        st.file_uploader('Upload your PDF or Web URL here & click on Process')
        st.button('Process')
    
if __name__ == '__main__':
    main()    