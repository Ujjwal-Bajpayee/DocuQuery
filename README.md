# 📄 DocuQuery

DocuQuery is a Streamlit-based document assistant that allows users to upload PDF documents, receive concise summaries, and interactively query the content using a Retrieval-Augmented Generation (RAG) pipeline powered by LangChain and Groq's LLMs.

## 🚀 Features

- 📤 Upload one or more PDF files
- 🧠 Summarize content using LLM (LLaMA3 via Groq)
- 💬 Ask natural language questions based on the uploaded documents
- 🔍 View source document excerpts for transparency

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq API](https://console.groq.com/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [FAISS](https://github.com/facebookresearch/faiss) for vector storage

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/docuquery.git
   cd docuquery

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt

5. **Create a .env file**
   ```bash
   GROQ_API_KEY=your_groq_api_key_here

6. **Run the app**
   ```bash
   streamlit run main.py

7. ** Project Structure**
├── main.py             # Streamlit app
├── rag_pipeline.py     # RAG pipeline logic (load, embed, retrieve, summarize)
├── .env                # Environment variable for API key (DO NOT SHARE)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
