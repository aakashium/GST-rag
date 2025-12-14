import streamlit as st
from rag_engine import get_rag_chain
import os

st.set_page_config(page_title="GST RAG Assistant", page_icon="📚")

st.title("📚 GST RAG Assistant")
st.markdown("Ask questions about GST Acts and Rules.")

# Check for API Key
if "GOOGLE_API_KEY" not in os.environ:
    st.warning("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG Chain 
@st.cache_resource
def load_chain():
    return get_rag_chain()  

try:
    qa_chain, retriever = load_chain()
except Exception as e:
    st.error(f"Error loading RAG chain: {e}")
    st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What is your question about GST?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = qa_chain.invoke(prompt)
                source_docs = retriever.invoke(prompt)
                
                st.markdown(answer)
                
                # Show sources in an expander
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1}:** Page {doc.metadata.get('page', 'N/A')}")
                        st.markdown(f"_{doc.page_content[:200]}..._")
                        st.divider()
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")
