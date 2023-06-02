import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import langchain
langchain.verbose = False
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub

def main():
    load_dotenv()

    st.set_page_config(page_title='Chat with Multiple PDFs', page_icon=':house:', layout='wide')
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with Multiple PDFs :books:')
    user_question = st.text_input('Ask a question about your documents:')
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader('Documents')
        pdf_docs = st.file_uploader('Upload your pdfs here:', type=['pdf'], accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing...'):
                # Get PDF Text 
                raw_text = get_pdf_texts(pdf_docs)

                # Get Text Chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                # Create Vector Store
                vector_store = get_vector_store(text_chunks)

                # Instance of a conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)



def handle_userinput(user_question):

    if st.session_state.conversation is None:
        return

    st.write(st.session_state.conversation)

    # st.write(st.session_state.conversation)
    # st.write(user_question)
    
    response = st.session_state.conversation({'question': user_question})
    st.write(response)
    
    # st.session_state.chat_history = response['chat_history']

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)


def get_conversation_chain(vector_store):

    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )

    return conversation_chain


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        texts=text_chunks, 
        embedding=embeddings)
    return vector_store


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks
    

def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


if __name__ == '__main__':
    main()