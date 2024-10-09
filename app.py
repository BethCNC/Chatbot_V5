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
    </style>
    """,
    unsafe_allow_html=True
)

# Debugging: Print environment variables to ensure they are correctly loaded
st.write(f"Using Pinecone Index: {os.getenv('PINECONE_INDEX')}")
st.write(f"Using Pinecone Environment: {os.getenv('PINECONE_ENV')}")
st.write(f"Using OpenAI API Key: {os.getenv('OPENAI_API_KEY')}")

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

# Load the Pinecone index
index = pinecone.Index(os.getenv('PINECONE_INDEX'))

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create Pinecone vector store
vectorstore = LangchainPinecone.from_existing_index(index_name=os.getenv('PINECONE_INDEX'), embedding=embeddings, namespace=os.getenv('PINECONE_NAMESPACE', ''))

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

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
st.markdown("### Chat History")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**User:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")

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
st.sidebar.image("assets/Disclaimer.png", width=50)
st.sidebar.warning("""
    **Disclaimer:** This chatbot is for educational purposes only. The information provided should not be considered medical advice. 
    Please consult with a healthcare professional for medical concerns.
""")