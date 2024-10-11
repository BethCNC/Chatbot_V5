import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

# Custom CSS (keep your existing CSS here)
st.markdown("""
<style>
    /* Your custom CSS */
</style>
""", unsafe_allow_html=True)

# App title
st.title('Chat with Dr. Spanos Ehler Danlos Research Articles')

# Set OpenAI LLM and embeddings
llm_chat = ChatOpenAI(temperature=0.3, max_tokens=150, model='gpt-4o-mini')
embeddings = OpenAIEmbeddings()

# Set Pinecone index
index_name = os.getenv("PINECONE_INDEX_NAME")
if index_name is None:
    raise ValueError("PINECONE_INDEX_NAME is not set in the environment variables")
docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm_chat,
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,  # Explicitly request source documents
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What do you want to know about EDS?"):
    # Display user message in chat message container
    with st.chat_message("user", avatar="AvatarZebra.png"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "AvatarZebra.png"})

    try:
        # Use the chain to get a response
        response = chain({"question": prompt})
        bot_response = response['answer']
        
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="AvatarDoctor.png"):
            st.markdown(bot_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response, "avatar": "AvatarDoctor.png"})

        # Display source documents
        with st.expander("Source Documents"):
            if 'source_documents' in response:
                for doc in response['source_documents']:
                    st.write(f"Source: {doc.metadata['source']}")
                    st.write(doc.page_content)
                    st.write("---")
            else:
                st.write("No source documents available for this response.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"Error details: {e}")