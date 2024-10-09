import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
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

# Custom CSS to use MabryPro font and set up the chat interface
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
    }
    
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    
    .stChat {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Load the Pinecone index
index_name = "spanos"
index = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create Pinecone vector store
vectorstore = Pinecone(index, embeddings.embed_query, "text")

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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know about EDS?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "assets/AvatarZebra.png"})
    # Display user message in chat message container
    with st.chat_message("user", avatar="assets/AvatarZebra.png"):
        st.markdown(prompt)
    
    # Generate AI response
    result = qa({"question": prompt, "chat_history": [(m["role"], m["content"]) for m in st.session_state.messages]})
    response = result["answer"]
    
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="assets/AvatarDoctor.png"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "assets/AvatarDoctor.png"})

# Display disclaimer
st.sidebar.image("assets/Disclaimer.png", width=50)
st.sidebar.markdown("""
    **Disclaimer:** This chatbot is for educational purposes only. The information provided should not be considered medical advice. 
    Please consult with a healthcare professional for medical concerns.
""")