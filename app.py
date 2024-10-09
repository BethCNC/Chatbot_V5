import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
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

# Custom CSS to use MabryPro font and align styles with Figma design
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

    /* Customize input box */
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

    /* Customize sidebar */
    .stSidebar {
        background-color: #FFF4F4;
        padding: 10px;
    }

    /* Disclaimer styling */
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

    /* Chat messages */
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

    /* Footer */
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

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-aws")

# Load the Pinecone index
index_name = "spanos"
index = pinecone.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create Pinecone vector store
vectorstore = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    text_key="text",
    namespace=""  # Use an empty string if you're not using a specific namespace
)

# Initialize ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Create ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Streamlit UI
st.title("Dr. Spanos EDS Chatbot")
st.subheader("This Chatbot is specifically trained on Dr. Spanos Research")

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">**User:** {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">**Assistant:** {message["content"]}</div>', unsafe_allow_html=True)

# Accept user input
user_input = st.text_input("What would you like to know about EDS?", key="user_input")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI response
    result = qa({"question": user_input, "chat_history": [(m["role"], m["content"]) for m in st.session_state.messages if m["role"] == "user"]})
    response = result["answer"]
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear the input field
    st.experimental_rerun()

# Display disclaimer in the sidebar
st.sidebar.image("assets/Disclaimer.png", width=35)  # Reduced the size
st.sidebar.warning("""
    **Disclaimer:** This chatbot is for educational purposes only. The information provided should not be considered medical advice. 
    Please consult with a healthcare professional for medical concerns.
""")

# Footer
st.markdown("<footer>Made with Streamlit</footer>", unsafe_allow_html=True)