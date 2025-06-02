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
    st.session_state['tell_more_state'] = {}
    st.session_state['detailed_query_contexts'] = []
    st.session_state['detailed_responses'] = {}

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

DETAILED_PROMPT_TEMPLATE = """<s>[INST]You are a medical chatbot trained on the latest data from HARRISON'S PRINCIPLES OF INTERNAL MEDICINE. For each section below, use bullet points or numbered lists to present information clearly and concisely.

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

# --- Per-message 'Tell me more' state and context ---
if 'tell_more_state' not in st.session_state:
    st.session_state['tell_more_state'] = {}
if 'detailed_query_contexts' not in st.session_state:
    st.session_state['detailed_query_contexts'] = []
if 'detailed_responses' not in st.session_state:
    st.session_state['detailed_responses'] = {}

# Display previous messages (only brief answers and user messages)
brief_message_count = 0  # Track the index of brief assistant messages
for i, message in enumerate(st.session_state.get("messages", [])):
    if message.get("role") == "assistant" and message.get("type") == "detailed":
        continue
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))
        if message.get("role") == "assistant" and message.get("type") == "brief":
            # Only allow 'Tell me more' if context exists for this brief message
            if brief_message_count < len(st.session_state['detailed_query_contexts']):
                if not st.session_state['tell_more_state'].get(brief_message_count, False):
                    if st.button("Tell me more...", key=f"tell_me_more_{brief_message_count}"):
                        st.session_state['tell_more_state'][brief_message_count] = True
                        st.rerun()
                elif st.session_state['tell_more_state'].get(brief_message_count, False):
                    # Check if we already have the detailed response stored
                    if brief_message_count in st.session_state['detailed_responses']:
                        # Display the stored detailed response
                        stored_response = st.session_state['detailed_responses'][brief_message_count]
                        st.markdown("⚠️ **_Note: Information provided is accordance to current medical diagnosis & treatment ._** ")
                        for title, section_content in stored_response['sections']:
                            with st.expander(title):
                                st.markdown(section_content)
                        if stored_response['recommended_professional']:
                            st.markdown(f"<div style='border:2px solid #ff6262; border-radius:10px; padding:16px; background:#fff0f0; margin-top:16px; margin-bottom:8px;'><b>Recommended Professional/Facility:</b> {stored_response['recommended_professional']}</div>", unsafe_allow_html=True)
                            if stored_response['resource_buttons']:
                                st.markdown("<div style='margin-top:12px; margin-bottom:8px;'><b>Find Nearby Services:</b></div>", unsafe_allow_html=True)
                                cols = st.columns(len(stored_response['resource_buttons']))
                                for j, (label, gmap_query) in enumerate(stored_response['resource_buttons']):
                                    gmap_url = f"https://www.google.com/maps/search/?api=1&query={gmap_query}"
                                    with cols[j]:
                                        st.markdown(f"""
                                            <a href='{gmap_url}' target='_blank' style='display:inline-block; margin:6px 0; color:#fff; background:linear-gradient(90deg,#ff6262,#ffd0d0); padding:12px 18px; border-radius:12px; text-decoration:none; font-weight:bold; font-size:1em; box-shadow:0 2px 8px #ffd0d0;'>
                                                {label}
                                            </a>
                                        """, unsafe_allow_html=True)
                                st.markdown(f"""
                                    <a href='https://snapwell.onrender.com/' target='_blank' style='display:inline-block; margin:6px 0; color:#fff; background:linear-gradient(90deg,#ff6262,#ffd0d0); padding:12px 18px; border-radius:12px; text-decoration:none; font-weight:bold; font-size:1em; box-shadow:0 2px 8px #ffd0d0;'>
                                        Book Appointment
                                    </a>
                                """, unsafe_allow_html=True)
                    else:
                        # Generate and store new detailed response
                        detailed_input = st.session_state['detailed_query_contexts'][brief_message_count]
                        detailed_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            memory=st.session_state.memory,
                            retriever=db_retriever,
                            combine_docs_chain_kwargs={'prompt': detailed_prompt}
                        )
                        result = detailed_chain.invoke(input=detailed_input["question"])
                        detailed_answer_raw = result["answer"]
                        # --- Improved section splitting and display ---
                        section_pattern = r"(\d+\. [^\n]+)"  # Matches section headers
                        matches = list(re.finditer(section_pattern, detailed_answer_raw))
                        sections = []
                        for idx2, match2 in enumerate(matches):
                            start = match2.start()
                            end = matches[idx2+1].start() if idx2+1 < len(matches) else None
                            section_text = detailed_answer_raw[start:end].strip()
                            lines = section_text.split('\n')
                            header = match2.group(0).strip()
                            if len(lines) > 1 and lines[1].strip().lower().startswith(header.split('.',1)[1].strip().lower()):
                                lines = [lines[0]] + lines[2:]
                            content_lines = [l for l in lines[1:] if l.strip()]
                            if content_lines:
                                sections.append((header, '\n'.join(content_lines)))
                        clarified_section_titles = [
                            "Overview",
                            "Common Symptoms & Signs",
                            "Diagnostic Procedures & Tests",
                            "Treatment Options",
                            "Lifestyle & Self-Care",
                            "Dietary Advice",
                            "Prognosis & Long-Term Management",
                            "Potential Complications & Prevention",
                            "When to Seek Emergency Care"
                        ]
                        
                        # Process sections for display and storage
                        processed_sections = []
                        for idx2, (header, section_content) in enumerate(sections):
                            lines = section_content.split('\n')
                            if sum(1 for l in lines if l.strip().startswith(('-', '*', '•', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'))) > 1:
                                formatted = '\n'.join([('- ' + l.lstrip('•- ')) if l.strip().startswith(('•', '-', '*')) else l for l in lines])
                                section_content = formatted
                            title = clarified_section_titles[idx2] if idx2 < len(clarified_section_titles) else header
                            processed_sections.append((title, section_content))
                        
                        # Extract recommended professional
                        recommended_professional = None
                        match = re.search(r"Recommended Professional/Facility:\s*(.*)", detailed_answer_raw, re.IGNORECASE)
                        if match:
                            recommended_professional = match.group(1).strip()
                        else:
                            # fallback: try to extract from last section
                            last_section = sections[-1][1] if sections else detailed_answer_raw
                            match2 = re.search(r"(physician|doctor|hospital|clinic|specialist|emergency|medical professional)", last_section, re.IGNORECASE)
                            if match2:
                                recommended_professional = match2.group(0).title()
                        
                        # Generate resource buttons
                        resource_buttons = []
                        if recommended_professional:
                            recs = re.split(r",| or | and ", recommended_professional)
                            filtered_recs = [rec.strip() for rec in recs if rec.strip() and len(rec.strip().split()) <= 5 and not any(x in rec.lower() for x in ["depending", "contact", "develop", "symptom", "seek", "provider", "action", "severity"])]
                            resource_buttons = [(rec, f"{rec.replace(' ', '+')}+near+me") for rec in filtered_recs]
                        
                        # Store the detailed response for future use
                        st.session_state['detailed_responses'][brief_message_count] = {
                            'sections': processed_sections,
                            'recommended_professional': recommended_professional,
                            'resource_buttons': resource_buttons
                        }
                        
                        # Display the response
                        st.markdown("⚠️ **_Note: Information provided is accordance to current medical diagnosis & treatment ._** ")
                        for title, section_content in processed_sections:
                            with st.expander(title):
                                st.markdown(section_content)
                        
                        if recommended_professional:
                            st.markdown(f"<div style='border:2px solid #ff6262; border-radius:10px; padding:16px; background:#fff0f0; margin-top:16px; margin-bottom:8px;'><b>Recommended Professional/Facility:</b> {recommended_professional}</div>", unsafe_allow_html=True)
                            if resource_buttons:
                                st.markdown("<div style='margin-top:12px; margin-bottom:8px;'><b>Find Nearby Services:</b></div>", unsafe_allow_html=True)
                                cols = st.columns(len(resource_buttons))
                                for j, (label, gmap_query) in enumerate(resource_buttons):
                                    gmap_url = f"https://www.google.com/maps/search/?api=1&query={gmap_query}"
                                    with cols[j]:
                                        st.markdown(f"""
                                            <a href='{gmap_url}' target='_blank' style='display:inline-block; margin:6px 0; color:#fff; background:linear-gradient(90deg,#ff6262,#ffd0d0); padding:12px 18px; border-radius:12px; text-decoration:none; font-weight:bold; font-size:1em; box-shadow:0 2px 8px #ffd0d0;'>
                                                {label}
                                            </a>
                                        """, unsafe_allow_html=True)
            brief_message_count += 1  # Increment counter for each brief assistant message

user_input = st.chat_input("WHAT CAN I ASSIST YOU FOR.....", key="chat_input")

# --- FIX: Always reset state and enforce correct flow on new query ---
if user_input and user_input != st.session_state.last_user_query:
    st.session_state.last_user_query = user_input
    st.session_state.show_detailed_response = False
    st.session_state.display_tell_more_button = False
    st.session_state.tell_more_button_clicked = False
    # Add new context for this query
    docs = db_retriever.get_relevant_documents(user_input)
    context_str = "\n\n".join([doc.page_content for doc in docs])
    detailed_context = {
        "context": context_str,
        "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"],
        "question": user_input
    }
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
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
            st.session_state['detailed_query_contexts'].append(detailed_context)
    st.rerun()

if st.session_state.get("messages"):
    st.button('Reset All Chat 🗑️', on_click=reset_conversation, key="reset_chat_button_global")

st.markdown('</div>', unsafe_allow_html=True)
