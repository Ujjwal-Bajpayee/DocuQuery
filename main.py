import streamlit as st
import tempfile
from dotenv import load_dotenv
from rag_pipeline import (
    load_and_split_docs,
    create_vectorstore,
    create_qa_chain,
    summarize_document,
)

load_dotenv()

st.set_page_config(page_title="DocuQuery", layout="wide")
st.title("📄 DocuQuery: LLM-Powered Document Assistant")

uploaded_files = st.file_uploader(
    "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        st.info(f"📂 Processing: {uploaded_file.name}")
        docs = load_and_split_docs(file_path)
        all_docs.extend(docs)

        full_text = "\n".join([doc.page_content for doc in docs])
        with st.spinner("🧠 Summarizing..."):
            summary = summarize_document(full_text)

        st.success("✅ Summary Generated")
        st.markdown(f"### 📝 Summary for **{uploaded_file.name}**\n\n{summary}")

    st.success("📚 All documents indexed. You can now ask questions.")
    vectorstore = create_vectorstore(all_docs)
    qa_chain = create_qa_chain(vectorstore)

    st.markdown("### 💬 Ask a question about the uploaded documents:")
    query = st.text_input("Enter your question:")

    if query and qa_chain:
        try:
            with st.spinner("Generating answer..."):
                result = qa_chain(query)
                st.subheader("📌 Answer")
                st.markdown(f"> {result['result']}")

                st.subheader("📚 Source Highlights")
                for i, doc in enumerate(result['source_documents']):
                    with st.expander(f"🔎 Source Chunk #{i+1}"):
                        st.markdown(doc.page_content)
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
            st.code(str(e))
