import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone as PineconeClient
import chainlit as cl

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

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
    return_source_documents=True,
)

@cl.on_chat_start
async def start():
    # Send a greeting message
    await cl.Message("Welcome! Chat with Dr. Spanos Ehler Danlos Research Articles. What do you want to know about EDS?").send()

    # Set avatars
    cl.user_session.set("doctor_avatar", cl.Avatar(path="assets/avatars/AvatarDoctor.png"))
    cl.user_session.set("user_avatar", cl.Avatar(path="assets/avatars/AvatarZebra.png"))

@cl.on_message
async def main(message: cl.Message):
    try:
        # Use the chain to get a response
        response = await cl.make_async(chain)({"question": message.content})
        bot_response = response['answer']
        
        # Send the bot response
        await cl.Message(content=bot_response, author="Dr. Spanos", avatar=cl.user_session.get("doctor_avatar")).send()

        # Display source documents
        if 'source_documents' in response:
            sources = cl.Text(content="Source Documents:")
            for doc in response['source_documents']:
                sources.content += f"\n\nSource: {doc.metadata['source']}\n{doc.page_content}\n---"
            await cl.Message(content="Here are the source documents:", elements=[sources]).send()
        else:
            await cl.Message(content="No source documents available for this response.").send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()
        print(f"Error details: {e}")