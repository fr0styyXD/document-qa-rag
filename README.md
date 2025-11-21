# 📚 Document Q&A with RAG

> An intelligent document question-answering system powered by Retrieval-Augmented Generation (RAG), enabling semantic search and natural language queries over PDF documents.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-1.0+-green.svg)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991.svg)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**🔗 [Live Demo](https://doc-q-a-rag.streamlit.app)** |

---

## 🎯 What is This?

A production-ready RAG (Retrieval-Augmented Generation) system that allows you to:
- 📄 **Upload PDF documents** (research papers, reports, books, manuals)
- 🤖 **Ask questions in natural language** (like talking to an expert)
- 💡 **Get accurate answers with source citations** (verify everything)
- 🔍 **Semantic search** (understands meaning, not just keywords)

### 🤔 The Problem

Large Language Models like ChatGPT have a knowledge cutoff. They can't answer questions about:
- ❌ Your proprietary documents
- ❌ Recent publications after their training
- ❌ Internal company knowledge bases
- ❌ Personal research papers

### ✨ The Solution: RAG

RAG (Retrieval-Augmented Generation) solves this by combining:
```
📄 Your Documents → 🔍 Semantic Search → 🤖 AI Generation
```

1. **Retrieval**: Find relevant information from YOUR documents using vector similarity
2. **Augmentation**: Add that information to the AI's context
3. **Generation**: AI generates accurate answers grounded in YOUR documents

---

## 🌟 Key Features

### For End Users:
- ✅ **Semantic Understanding** - Finds answers by meaning, not just keywords
- ✅ **Source Citations** - Every answer shows WHERE the information came from
- ✅ **Multi-Document Support** - Upload and query multiple PDFs simultaneously
- ✅ **Natural Language** - Ask questions like you're talking to a person
- ✅ **Fast & Efficient** - Cached embeddings for instant subsequent queries

### For Developers:
- ✅ **Production-Ready** - Comprehensive error handling and logging
- ✅ **Modular Architecture** - Easy to extend and customize
- ✅ **Modern Stack** - LangChain 1.0+ with LCEL (LangChain Expression Language)
- ✅ **Well-Documented** - Every function has detailed explanations
- ✅ **Cost-Optimized** - Smart caching reduces API costs by 90%

---

## 🏗️ Architecture

### High-Level Flow
```
┌─────────────────────────────────────────────────────────┐
│                  USER UPLOADS PDF                        │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│          DOCUMENT PROCESSING PIPELINE                    │
│                                                          │
│  PDF → Extract Text → Chunk (512 tokens)                │
│           ↓                                              │
│  Generate Embeddings (OpenAI - 1536 dims)               │
│           ↓                                              │
│  Store in FAISS Vector Database                         │
│           ↓                                              │
│  Cache to Disk                                          │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│              USER ASKS QUESTION                          │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│              RAG QUERY PIPELINE                          │
│                                                          │
│  Question → Embed → Search FAISS (Cosine Similarity)    │
│           ↓                                              │
│  Retrieve Top 4 Most Similar Chunks                     │
│           ↓                                              │
│  Build Prompt: Context + Question                       │
│           ↓                                              │
│  Send to GPT-3.5-turbo                                  │
│           ↓                                              │
│  Generate Answer + Return Sources                       │
└─────────────────────────────────────────────────────────┘
```

### Technical Architecture
```python
# LangChain 1.0 LCEL (LangChain Expression Language)
rag_chain = (
    {
        "context": retriever | format_docs,  # Retrieve & format documents
        "question": RunnablePassthrough()     # Pass question through
    }
    | prompt              # Build prompt
    | llm                # Send to GPT-3.5
    | StrOutputParser()   # Parse response
)
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | LangChain 1.0.7 | RAG pipeline orchestration with LCEL |
| **Embeddings** | OpenAI text-embedding-3-small | Text → 1536-dimensional vectors |
| **LLM** | OpenAI GPT-3.5-turbo | Natural language generation |
| **Vector DB** | FAISS (Facebook AI) | Fast similarity search (millisecond queries) |
| **UI Framework** | Streamlit 1.40.2 | Interactive web interface |
| **PDF Processing** | PyPDF 5.1.0 | Text extraction from PDFs |
| **Language** | Python 3.8+ | Core implementation |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/document-qa-rag.git
cd document-qa-rag

# 2. Create virtual environment
python -m venv venv

# Activate virtual environment:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file (for local development)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 5. Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage

### Step 1: Provide API Key

When you first open the app:
- If running locally with `.env`: API key is loaded automatically
- If running deployed version: Enter your OpenAI API key in the UI

**Don't have an API key?**
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up (free)
3. Add $5 minimum credit
4. Create API key

### Step 2: Upload Documents

1. Click **"Choose PDF files"** in the sidebar
2. Select one or more PDF documents
3. Click **"🚀 Process Documents"**
4. Wait 10-60 seconds for processing

The system will:
- Extract text from PDFs
- Split into 512-token chunks
- Generate embeddings (~$0.02 per 100 pages)
- Store in vector database
- Cache for future use

### Step 3: Ask Questions

1. Type your question in the input box
2. Click **"🔍 Get Answer"**
3. View answer with source citations
4. Click on source expandable sections to verify

### Example Questions

**Research Papers:**
- "What are the main findings of this study?"
- "What methodology was used?"
- "What are the limitations mentioned?"
- "Compare approach X with approach Y"

**Business Documents:**
- "What were the Q3 revenue figures?"
- "Summarize the key strategic initiatives"
- "What risks are identified?"

**Technical Documentation:**
- "How do I configure feature X?"
- "What are the system requirements?"
- "Explain the authentication flow"

---

## 💰 Cost Estimates

### Per Document (average 10-page PDF):

| Operation | Cost |
|-----------|------|
| Embedding generation | $0.001 - $0.002 |
| Storage (local FAISS) | Free |
| **Total per document** | **~$0.002** |

### Per Question:

| Operation | Cost |
|-----------|------|
| Question embedding | $0.000002 |
| LLM generation (GPT-3.5) | $0.001 - $0.005 |
| **Total per question** | **~$0.002** |

### Real-World Example:

**100-page research paper + 50 questions:**
- Document processing: $0.02
- 50 questions: $0.10
- **Total: $0.12** (12 cents!)

**Very affordable for most use cases!** 🎉

---

## 🎯 Key Design Decisions

### Why These Choices?

#### 1. **Chunk Size: 512 tokens**

**Too small (128 tokens):**
- ❌ Loses context
- ❌ Sentences split mid-thought
- ❌ Poor answer quality

**Too large (2048 tokens):**
- ❌ Less precise retrieval
- ❌ More expensive
- ❌ Context window fills up

**512 tokens = Industry standard sweet spot** ✅

#### 2. **Top-K Retrieval: 4 documents**

**Too few (1-2):**
- ❌ Not enough context
- ❌ Might miss relevant info

**Too many (10+):**
- ❌ Adds noise
- ❌ More expensive
- ❌ Can confuse LLM

**4 documents = ~2000 tokens of context** ✅

#### 3. **GPT-3.5-turbo vs GPT-4**

| Model | Cost | Speed | Best For |
|-------|------|-------|----------|
| GPT-3.5-turbo | $0.001/1K | ~500ms | Q&A, summaries |
| GPT-4 | $0.03/1K | ~2s | Complex reasoning |

**For RAG Q&A: GPT-3.5 is 30x cheaper and sufficient!** ✅

#### 4. **FAISS vs Pinecone/Weaviate**

| Vector DB | Cost | Speed | Best For |
|-----------|------|-------|----------|
| FAISS | Free | Very fast | Demos, prototypes |
| Pinecone | $70+/mo | Fast | Production (large scale) |
| Weaviate | Self-host | Fast | Production (self-hosted) |

**For portfolios/demos: FAISS is perfect!** ✅

---

## 🔧 Configuration

Edit these constants in `app.py` to customize:
```python
# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's latest

# LLM for generation
LLM_MODEL = "gpt-3.5-turbo"  # Fast & cheap
LLM_TEMPERATURE = 0           # Factual (0) vs Creative (1)

# Document processing
CHUNK_SIZE = 512      # Tokens per chunk
CHUNK_OVERLAP = 50    # Overlap prevents splitting

# Retrieval
TOP_K_DOCUMENTS = 4   # Number of chunks to retrieve

# Storage
VECTOR_STORE_PATH = "data/faiss_index"  # Cache location
```

---

## 📁 Project Structure
```
document-qa-rag/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                     # API keys (local only - NOT committed)
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── LICENSE                  # MIT License
├── documents/               # Uploaded PDFs (temporary)
│   └── .gitkeep
├── data/                    # Vector store cache
│   └── faiss_index/
│       ├── index.faiss      # FAISS index
│       └── index.pkl        # Document metadata
└── screenshots/             # App screenshots for README
    ├── upload.png
    ├── processing.png
    └── qa_demo.png
```

---

## 🧠 What I Learned

### Technical Skills:
- **RAG Architecture**: Understanding retrieval-augmented generation from first principles
- **Vector Embeddings**: How text is converted to semantic vectors
- **Semantic Search**: Cosine similarity and vector databases
- **LangChain LCEL**: Modern chain composition with pipe operators
- **Prompt Engineering**: Crafting prompts that prevent hallucinations
- **API Integration**: OpenAI embeddings and chat completion APIs
- **Streamlit**: Building interactive ML applications

### Engineering Decisions:
- **Chunk Size Optimization**: Balancing context vs precision (tested 128, 512, 1024, 2048)
- **Cost Optimization**: Implementing caching to reduce API costs by 90%
- **Error Handling**: Production-ready error messages and graceful degradation
- **User Experience**: Loading indicators, progress bars, source citations

### Challenges Overcome:
- **LangChain 1.0 Migration**: Adapted to breaking changes in LangChain 1.0
- **Windows Compatibility**: Resolved `pwd` module issues on Windows
- **Version Conflicts**: Managed complex dependency tree across langchain packages
- **Memory Management**: Efficient document processing for large PDFs

---

## 🚀 Future Enhancements

### Planned Features:
- [ ] **Multi-format support**: DOCX, TXT, HTML, Markdown
- [ ] **Conversation memory**: Multi-turn conversations with context
- [ ] **Advanced retrieval**: Hybrid search (dense + sparse)
- [ ] **Reranking**: Score and rerank retrieved chunks
- [ ] **Streaming responses**: Token-by-token answer streaming
- [ ] **Document management**: Upload, delete, update documents
- [ ] **User authentication**: Multi-user support with separate indexes
- [ ] **Analytics dashboard**: Query stats, popular questions
- [ ] **Export functionality**: Save Q&A sessions as PDF/Markdown
- [ ] **GPT-4 option**: Toggle between GPT-3.5 and GPT-4

### Deployment Options:
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] API endpoint (FastAPI wrapper)

---

## 📧 Contact

**Aksh Dhingra**
- GitHub: [@fr0styyXD](https://github.com/fr0styyXD)
- LinkedIn: [Aksh Dhingra](https://linkedin.com/in/akshdhingra)
- Email: workdhingra26@gmail.com

---

[⬆ Back to Top](#-document-qa-with-rag)

</div>
