from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time
import re
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    st.error("TOGETHER_API_KEY not found in .env. Please set it.")
    st.stop()

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
    st.session_state.last_user_query = ""
    st.session_state.show_detailed_response = False
    st.session_state.last_detailed_query_context = None
    st.session_state.display_tell_more_button = False
    st.session_state.tell_more_button_clicked = False

# Session State Variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""
if "show_detailed_response" not in st.session_state:
    st.session_state.show_detailed_response = False
if "last_detailed_query_context" not in st.session_state:
    st.session_state.last_detailed_query_context = None
if "display_tell_more_button" not in st.session_state:
    st.session_state.display_tell_more_button = False
if "tell_more_button_clicked" not in st.session_state:
    st.session_state.tell_more_button_clicked = False

# Set up embeddings for vector search
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})

# Load the FAISS vector database
db = FAISS.load_local("./ipc_vector_db_med", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the two prompt templates
BRIEF_PROMPT_TEMPLATE = """<s>[INST]You are a medical chatbot. Given the context, provide a brief and informative answer to the user's question. Focus on delivering the most essential information, such as a definition, common cause, main symptom, or primary recommendation, in 2-4 concise sentences.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
BRIEF ANSWER:
</s>[INST]
"""

DETAILED_PROMPT_TEMPLATE = """<s>[INST]You are a medical chatbot trained on the latest data from HARRISON'S PRINCIPLES OF INTERNAL MEDICINE. Provide comprehensive information including:
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

At the end of your response, after addressing the above points, provide a clear recommendation for the type of medical professional or facility the user should consult for this condition. Format this recommendation as: "Recommended Professional/Facility: [Type of Doctor/Facility, e.g., General Physician, Dermatologist, Cardiologist, Hospital, Clinic]"

Ensure your responses are professional, concise, and aligned with established medical standards and guidelines. Prioritize the user's specific query while providing a well-rounded answer. If the question falls outside your knowledge base or requires personalized medical advice, recommend consulting a healthcare professional.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
DETAILED ANSWER:
</s>[INST]
"""

brief_prompt = PromptTemplate(template=BRIEF_PROMPT_TEMPLATE, input_variables=['context', 'question', 'chat_history'])
detailed_prompt = PromptTemplate(template=DETAILED_PROMPT_TEMPLATE, input_variables=['context', 'question', 'chat_history'])

# Set up the language model (LLM)
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=TOGETHER_API_KEY
)

# Helper function to generate and stream response with typing effect
def generate_and_stream_response(chain, input_to_chain, message_placeholder):
    full_response = "⚠️ **_Note: Information provided is accordance to current medical diagnosis & treatment ._** \n\n\n"
    
    result = chain.invoke(input=input_to_chain)
    answer_text = result["answer"]

    for i in range(len(answer_text)):
        time.sleep(0.02)
        message_placeholder.markdown(full_response + answer_text[:i+1] + " ▌")
    message_placeholder.markdown(full_response + answer_text)
    return answer_text

# Helper function to generate and stream PRE-COMPOSED response
def stream_text_with_typing_effect(message_placeholder, text_to_stream):
    for i in range(len(text_to_stream)):
        time.sleep(0.02)
        message_placeholder.markdown(text_to_stream[:i+1] + " ▌")
    message_placeholder.markdown(text_to_stream)

# Display previous messages
for i, message in enumerate(st.session_state.get("messages", [])):
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))
        if message.get("role") == "assistant" and \
           message.get("type") == "brief" and \
           st.session_state.display_tell_more_button and \
           i == len(st.session_state.messages) - 1:
            if st.button("Tell me more...", key=f"tell_me_more_{i}"):
                st.session_state.tell_more_button_clicked = True
                st.session_state.display_tell_more_button = False
                st.rerun()

user_input = st.chat_input("WHAT CAN I ASSIST YOU FOR.....", key="chat_input")


# Handling new user input and brief response
if user_input and user_input != st.session_state.last_user_query and not st.session_state.tell_more_button_clicked:
    st.session_state.last_user_query = user_input 
    st.session_state.show_detailed_response = False
    st.session_state.display_tell_more_button = False

    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    docs = db_retriever.get_relevant_documents(user_input)
    context_str = "\n\n".join([doc.page_content for doc in docs])

    st.session_state.last_detailed_query_context = {
        "context": context_str,
        "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"],
        "question": user_input
    }

    with st.chat_message("assistant"):
        with st.status("Introspecting 💡...", expanded=True):
            brief_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                memory=st.session_state.memory,
                retriever=db_retriever,
                combine_docs_chain_kwargs={'prompt': brief_prompt}
            )
            brief_answer = generate_and_stream_response(brief_chain, user_input, st.empty())
            
            st.session_state.messages.append({"role": "assistant", "content": brief_answer, "type": "brief", "original_query": user_input})

        st.session_state.display_tell_more_button = True
        st.session_state.tell_more_button_clicked = False
        st.rerun()

# Handling detailed response
if st.session_state.tell_more_button_clicked and st.session_state.last_detailed_query_context is not None:
    st.session_state.show_detailed_response = True
    st.session_state.tell_more_button_clicked = False

    detailed_input = st.session_state.last_detailed_query_context

    with st.chat_message("assistant"):
        
        with st.status("Fetching more details...", expanded=True):
            message_placeholder = st.empty()

            detailed_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                memory=st.session_state.memory,
                retriever=db_retriever,
                combine_docs_chain_kwargs={'prompt': detailed_prompt}
            )

            result = detailed_chain.invoke(input=detailed_input["question"])
            detailed_answer_raw = result["answer"]

            recommended_professional = None
            match = re.search(r"Recommended Professional/Facility:\s*(.*)", detailed_answer_raw, re.IGNORECASE)
            if match:
                recommended_professional = match.group(1).strip()
            else:
                if "general physician" in detailed_answer_raw.lower() or "doctor" in detailed_answer_raw.lower():
                    recommended_professional = "General Physician"
                elif "hospital" in detailed_answer_raw.lower() or "clinic" in detailed_answer_raw.lower() or "emergency" in detailed_answer_raw.lower():
                    recommended_professional = "Hospital"
                if not recommended_professional:
                    recommended_professional = "Medical Professional"

            final_detailed_message_content = "⚠️ **_Note: Information provided is accordance to current medical diagnosis & treatment ._** \n\n\n" + detailed_answer_raw

            if recommended_professional:
                search_term = recommended_professional.replace(" ", "+") + "+near+me"
                Maps_url = f"https://www.google.com/maps/search/?api=1&query={search_term}"
                final_detailed_message_content += f"\n\n**Find a {recommended_professional} near you:** [Search on Google Maps]({Maps_url})"

            stream_text_with_typing_effect(message_placeholder, final_detailed_message_content)
        
        st.session_state.messages.append({"role": "assistant", "content": final_detailed_message_content, "type": "detailed"})

    st.session_state.last_detailed_query_context = None
    st.session_state.show_detailed_response = False

if st.session_state.get("messages"):
    st.button('Reset All Chat 🗑️', on_click=reset_conversation, key="reset_chat_button_global")

st.markdown('</div>', unsafe_allow_html=True)