import streamlit as st
import os
from rag_engine import MitigationAgent

st.set_page_config(
    page_title="Context-Aware Hallucination Mitigation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Safe custom CSS */
    .fact-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .badge-verified {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
    }
    .badge-warning {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fb7185;
        border: 1px solid #ef4444;
    }
    .main-title {
        background: linear-gradient(to right, #4ade80, #3b82f6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Context-Aware Hallucination Mitigation System</div>', unsafe_allow_html=True)

if "agent" not in st.session_state:
    try:
        st.session_state.agent = MitigationAgent()
        st.session_state.db_error = None
    except Exception as e:
        st.session_state.agent = None
        st.session_state.db_error = str(e)

agent = st.session_state.agent

if st.session_state.db_error:
    st.error("⚠️ Offline Mode: Database connection refused. Please start Endee Vector DB.")

with st.sidebar:
    st.markdown("### 📥 Knowledge Upload")
    doc_upload = st.file_uploader("Upload reference documents", type=["txt"])
    
    if st.button("Index Content", disabled=agent is None, use_container_width=True, type="primary"):
        if agent is None:
            st.error("Cannot index while offline.")
        elif doc_upload:
            text = doc_upload.getvalue().decode('utf-8')
            with st.spinner("Processing..."):
                agent.load_document(text)
            st.success("Indexing complete.")
        else:
            st.warning("Please upload a file.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "status" in msg:
            if "HALLUCINATION" in msg["status"]:
                st.markdown("<div class='fact-badge badge-warning'>⚠️ Potential Inaccuracy Detected</div>", unsafe_allow_html=True)
            elif "FACTUAL" in msg["status"]:
                st.markdown("<div class='fact-badge badge-verified'>✓ Verified Fact</div>", unsafe_allow_html=True)

# Chat Input Flow
if prompt := st.chat_input("Enter your prompt...", disabled=agent is None):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if agent:
            response, status, context = agent.generate_and_verify(prompt)
            st.markdown(response)
            
            if "HALLUCINATION" in status:
                st.markdown("<div class='fact-badge badge-warning'>⚠️ Potential Inaccuracy Detected</div>", unsafe_allow_html=True)
            elif "FACTUAL" in status:
                st.markdown("<div class='fact-badge badge-verified'>✓ Verified Fact</div>", unsafe_allow_html=True)
                
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "status": status
            })
