import streamlit as st
import os
from rag_engine import MitigationAgent

st.set_page_config(
    page_title="Context-Aware Hallucination Mitigation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-1d391kg {
        background-color: #1a1c23;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        background-color: #2b2e35;
        color: white;
        border-radius: 5px;
        border: 1px solid #4CAF50;
    }
    h1, h2, h3 {
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

if "agent" not in st.session_state:
    try:
        st.session_state.agent = MitigationAgent()
        st.session_state.db_error = None
    except Exception as e:
        st.session_state.agent = None
        st.session_state.db_error = str(e)

agent = st.session_state.agent

st.title("Context-Aware Hallucination Mitigation System")
st.markdown("Powered by **Endee Vector Database** for sub-millisecond similarity search and real-time contextual grounding.")

if st.session_state.db_error:
    st.error(f"⚠️ Backend Offline: {st.session_state.db_error}")

with st.sidebar:
    st.header("Knowledge Ingestion")
    st.markdown("Upload documents to build ground truth.")
    doc_upload = st.file_uploader("Upload Ground Truth Document", type=["txt"])
    if st.button("Process Document", disabled=agent is None):
        if agent is None:
            st.error("Cannot process: Database offline.")
        elif doc_upload:
            text = doc_upload.getvalue().decode('utf-8')
            agent.load_document(text)
            st.success("Document successfully indexed in Vector DB.")
        else:
            st.warning("Please upload a document first.")

st.markdown("### LLM Interaction Panel")
query = st.text_input("Enter your prompt for the LLM:")

if st.button("Generate & Verify", disabled=agent is None):
    if agent is None:
        st.error("Cannot generate: Database offline.")
    elif not query:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Processing request through pipeline..."):
            response, status, context = agent.generate_and_verify(query)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Model Response")
                st.info(response)
                
                with st.expander("View Retrieved Context"):
                    st.write(context if context else "No context found.")
            
            with col2:
                st.markdown("#### Verification Status")
                if "HALLUCINATION" in status:
                    st.error(f"Status: {status} ⚠️")
                elif "FACTUAL" in status:
                    st.success(f"Status: {status} ✅")
                else:
                    st.warning(f"Status: {status}")
