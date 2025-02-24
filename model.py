# Import necessary libraries
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time
from dotenv import load_dotenv # load specific environment that been created
 
load_dotenv()
## Langsmith project tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Set up the Streamlit page configuration
st.set_page_config(page_title="MEDCHAT", layout="wide")

st.markdown(
    """
    <style>
    /* Main container for flexbox layout */
    .main {
        display: flex;
    }
    
    /* Sidebar styling */
    .sidebar {
        width: 300px;
        padding: 20px;
        height: 100vh;
        position: fixed;
        background-color: #000000;
        left: 0;
        top: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    /* Main chat container styling */
    .chat-container {
        flex: 1;
        padding: 20px;
        margin-left: 300px;
    }
    
    .stApp, .ea3mdgi6 {
        background-color: #000000; /* right side bg color */
    }
    
    div.stButton > button:first-child {
        background-color: #ffd0d0;
    }
    div.stButton > button:active {
        background-color: #ff6262;
    }
    
    div[data-testid="stStatusWidget"] div button {
        display: none;
    }
    
    /* Adjust top margin of the report view container */
    .reportview-container {
        margin-top: -2em;
    }
    
    /* Hide various Streamlit elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    
    /* Ensure the placeholder text is also visible */
    .stTextInput > div > div > input::placeholder {
        color: #666666 !important;
    }
    
    .stChatMessage {
        background-color: #28282B; /* chat message background color set to black */
        color : #000000 !important;
    }


    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("med-bot.svg", width=290)
    st.title("MEDICHAT")
    st.markdown("Your AI MEDICAL ASSISTANT")

# Main chat interface container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Function to reset the conversation
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize conversation memory
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

# Set up embeddings for vector search
embedings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True,"revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})

# Load the FAISS vector database
# db = FAISS.load_local("./ipc_vector_db_cmdt", embedings, allow_dangerous_deserialization=True)
db = FAISS.load_local("./ipc_vector_db_med", embedings, allow_dangerous_deserialization=True)

db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

# Define the prompt template for the AI
# THIS IS ACTUALLY TELLING CHATBOT WHAT U ARE AND CHAIN WHAT U HAVE TO DO
# FOR CMDT-2023 DATA
# prompt_template = """<s>[INST]You are a medical chatbot trained on the latest data in diagnosis and treatment, designed to provide accurate and concise information in response to users' medical queries. Your primary focus is to offer evidence-based answers related to symptoms, infections, disorders, diseases, and their respective treatments. Refrain from generating hypothetical diagnoses or questions, and stick strictly to the context provided. Ensure your responses are professional, concise, and relevant. If the question falls outside the given context, do not rely on chat history; instead, generate an appropriate response based on your medical knowledge. Prioritize the user's query, avoid unnecessary details, and ensure compliance with medical standards and guidelines.
# CONTEXT: {context}
# CHAT HISTORY: {chat_history}
# QUESTION: {question}
# ANSWER:
# </s>[INST]
# """

## FOR MEDICAL DATA
prompt_template = """<s>[INST]You are a medical chatbot trained on the latest data in diagnosis and treatment from HARRISON'S PRINCIPLES OF INTERNAL MEDICINE and other authoritative medical sources. Your primary focus is to provide accurate, evidence-based information related to medical conditions and their management. When presented with a query about a specific disease or condition, provide comprehensive information including:

1. Brief overview of the condition
2. Common symptoms and signs
3. Diagnostic procedures and tests
4. Treatment options:
   a. Medications (including dosages and potential side effects)
   b. Surgical interventions (if applicable)
   c. Other therapeutic approaches
5. Lifestyle modifications and self-care measures
6. Dietary recommendations and restrictions
7. Prognosis and long-term management
8. Potential complications and how to prevent them
9. When to seek immediate medical attention

Ensure your responses are professional, concise, and aligned with established medical standards and guidelines. Prioritize the user's specific query while providing a well-rounded answer. If the question falls outside your knowledge base or requires personalized medical advice, recommend consulting a healthcare professional.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

# Create a PromptTemplate object
prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

# Set up the language model (LLM)
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key="3057a359ca444b31fd81b6a0958283873ebce147defd0d827d45330e98a536a6"
)

# Create the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True),
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display previous messages
for message in st.session_state.get("messages", []):
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("WHAT CAN I ASSIST YOU FOR.....")#input text box for user to ask question

# Handle user input
if input_prompt:
    # Display user message
    with st.chat_message("user"):
        st.write(input_prompt)

    # Add user message to session state
    st.session_state.messages.append({"role":"user","content":input_prompt})

    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.status("Introspecting 💡...",expanded=True):
            # Invoke the QA chain to get the response
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided is accordance to current medical diagnosis & treatment ._** \n\n\n"
        # Stream the response
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        # Add a button to reset the conversation
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    # Add AI response to session state
    st.session_state.messages.append({"role":"assistant","content":result["answer"]})

# Close the chat container div
st.markdown('</div>', unsafe_allow_html=True)