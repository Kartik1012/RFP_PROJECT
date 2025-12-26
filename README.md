# RFP_PROJECT


📄 RFP RAG Pipeline (with Source Citations)

A production-ready Retrieval-Augmented Generation (RAG) pipeline to automatically extract questions from RFP documents, generate context-grounded answers, and produce a professional PDF response with file- and page-level citations.

✨ Key Features

📥 Ingest PDFs / DOCX / TXT with metadata (file name, page)

❓ Automatically extract RFP questions using LLMs

🧠 FAISS-based semantic retrieval with OpenAI embeddings

📝 High-quality answers generated strictly from retrieved context

📌 Citations shown only at the end (no inline references)

📄 Professional PDF output (Q&A + Sources)

🔁 Backward-compatible FAISS index loading

🧪 Comprehensive test suite

🏗️ High-Level Architecture
Documents
   ↓
DocumentProcessor (chunking + metadata)
   ↓
FAISS Vector Store (text + metadata)
   ↓
Retriever
   ↓
QA Engine (RAG)
   ↓
PDF Generator

📂 Project Structure
```
.
├── config.py
├── document_processor.py
├── question_extractor.py
├── vector_store_manager.py   # FAISS + Retriever (merged file)
├── qa_engine.py
├── pdf_generator.py
├── main.py                   # Pipeline entry point
├── test.py                   # Test suite
├── internal/                 # Knowledge base documents
├── input/                    # RFP documents
└── output/                   # Generated PDFs
```
⚙️ Setup (Windows)
1️⃣ Create & activate virtual environment
```
python -m venv venv
venv\Scripts\activate
```

2️⃣ Install dependencies
```
pip install -r requirements.txt
```
3️⃣ 
## 3️⃣ ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

▶️ Usage
🔹 Build index + answer RFP
python main.py path\to\rfp.pdf --rebuild-index

🔹 Output

Answered questions

End-only citations (file + page)

PDF saved to output/

🧪 Run Tests
```
python test.py
```
🧪 Test Coverage

The automated test suite validates the complete pipeline, including:

✅ Configuration loading

📄 Document ingestion and parsing

🧠 Vector store creation and retrieval

✍️ Question–Answer generation

🔁 End-to-end pipeline execution

 
📌 Citation Format

Sources (end of response only):
```
[1] Policy_Document.pdf, Page 4
[2] Claims_SOP.pdf, Page 6
```

No inline citations are used.

🔄 Important Note (FAISS Index)

If upgrading from an older version:
```
del faiss_index.pkl
python main.py --rebuild-index
```


This ensures metadata (file name, page) is stored correctly.

🚀 Future Enhancements

1. Confidence scoring per answer

2. Source deduplication

3. REST API deployment

4. UI integration

5. Streaming responses
