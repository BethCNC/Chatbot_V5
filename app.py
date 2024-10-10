import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Dr. Spanos EDS Chatbot",
    page_icon="assets/favicon.ico",
    layout="wide"
)

# Custom CSS (unchanged)
st.markdown(
    """
    <style>
    @font-face {
        font-family: 'MabryPro';
        src: url('assets/MabryPro-Regular.ttf') format('truetype');
        font-weight: normal;
        font-style: normal;
    }
    
    * {
        font-family: 'MabryPro', sans-serif;
        color: #4A4A4A;
    }

    .stTextInput label {
        font-size: 18px;
        color: #FF5A7C;
    }
    .stTextInput input {
        background-color: #F6F6F6;
        border-radius: 12px;
        border: 1px solid #FF5A7C;
        padding: 10px;
        font-size: 16px;
    }

    .stSidebar {
        background-color: #FFF4F4;
        padding: 10px;
    }

    .stSidebar .sidebar-content .stImage {
        width: 35px;
        height: 35px;
    }

    .stSidebar .sidebar-content .stAlert {
        background-color: #FFF4F4;
        color: #FF5A7C;
        border-left: 6px solid #FF5A7C;
        font-size: 14px;
    }

    .user-message {
        background-color: #FFEBF0;
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    .assistant-message {
        background-color: #F0F8FF;
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    footer {
        font-size: 12px;
        text-align: center;
        margin-top: 50px;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def initialize_pinecone():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX")

    if not all([pinecone_api_key, pinecone_index_name]):
        raise ValueError("Missing Pinecone environment variables. Please check your .env file.")

    pc = Pinecone(api_key=pinecone_api_key)
    if pinecone_index_name is None:
        raise ValueError("PINECONE_INDEX environment variable is not set")
    return pc.Index(pinecone_index_name)

@st.cache_resource
def initialize_qa_chain():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX")

    if not all([openai_api_key, pinecone_api_key, pinecone_index_name]):
        raise ValueError("Missing environment variables. Please check your .env file.")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    
    vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")
    
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

def main():
    st.title("Dr. Spanos EDS Chatbot")
    st.subheader("This Chatbot is specifically trained on Dr. Spanos Research")

    # Initialize Pinecone and QA chain
    try:
        index = initialize_pinecone()
        qa = initialize_qa_chain()
    except Exception as e:
        st.error(f"Error initializing the application: {e}")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        st.markdown(f'<div class="{message["role"]}-message">**{role}:** {message["content"]}</div>', unsafe_allow_html=True)

    # User input
    user_input = st.text_input("What would you like to know about EDS?", key="user_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Generating response..."):
            result = qa({"question": user_input, "chat_history": [(m["role"], m["content"]) for m in st.session_state.messages if m["role"] == "user"]})
        
        response = result["answer"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Sidebar
    st.sidebar.image("assets/Disclaimer.png", width=35)
    st.sidebar.warning("""
        **Disclaimer:** This chatbot is for educational purposes only. The information provided should not be considered medical advice. 
        Please consult with a healthcare professional for medical concerns.
    """)

    # Footer
    st.markdown("<footer>Made with Streamlit</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()