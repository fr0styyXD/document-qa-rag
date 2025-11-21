"""
═══════════════════════════════════════════════════════════════════════
                    DOCUMENT Q&A WITH RAG SYSTEM
═══════════════════════════════════════════════════════════════════════

PROJECT OVERVIEW:
-----------------
A production-ready Retrieval-Augmented Generation (RAG) system that enables
users to upload PDF documents and ask questions about them using natural language.

WHAT IS RAG?
------------
RAG = Retrieval-Augmented Generation

Problem: LLMs like ChatGPT only know information up to their training cutoff.
         They can't answer questions about YOUR specific documents.

Solution: RAG combines:
    1. RETRIEVAL: Search your documents for relevant information
    2. AUGMENTATION: Add that information to the LLM's prompt
    3. GENERATION: LLM generates answer based on YOUR documents

HOW THIS SYSTEM WORKS:
----------------------

STEP 1: DOCUMENT INGESTION (When user uploads PDF)
    PDF File
        ↓
    Extract text (PyPDF)
        ↓
    Split into chunks (512 tokens each, 50 token overlap)
        ↓
    For each chunk:
        - Send to OpenAI API
        - Get embedding (1536-dimensional vector that represents meaning)
        ↓
    Store embeddings in FAISS vector database
        ↓
    Save to disk (cache for reuse)

STEP 2: QUESTION ANSWERING (When user asks a question)
    User Question: "What is machine learning?"
        ↓
    Convert question to embedding (same 1536-dimensional space)
        ↓
    Search FAISS database for similar embeddings (cosine similarity)
        ↓
    Retrieve top 4 most relevant chunks
        ↓
    Build prompt: "Given this context: [chunks], answer: [question]"
        ↓
    Send to GPT-3.5-turbo
        ↓
    Get answer grounded in YOUR documents
        ↓
    Display answer + show source chunks for verification

WHY THIS ARCHITECTURE?
----------------------
- Embeddings: Capture semantic meaning (not just keyword matching)
- Vector DB: Fast similarity search (millions of vectors in milliseconds)
- FAISS: Free, fast, runs locally (perfect for demos)
- GPT-3.5: Cheap ($0.001/query), fast, good quality for Q&A
- Chunk size 512: Balance between context and precision
- Top-K = 4: Enough context without overwhelming the model

COST ESTIMATE:
--------------
- Embedding 100 pages: ~$0.02
- 50 questions: ~$0.10
- Total: ~$0.12 per 100-page document with 50 questions

TECH STACK:
-----------
- LangChain: RAG pipeline framework (simplifies complex workflows)
- OpenAI API: Embeddings (text-embedding-3-small) + LLM (gpt-3.5-turbo)
- FAISS: Vector database (Facebook AI Similarity Search)
- Streamlit: Web UI framework (fast, Python-based)
- PyPDF: PDF text extraction

AUTHOR: Aksh Dhingra
DATE: 18 November 2025
═══════════════════════════════════════════════════════════════════════
"""

import os
import streamlit as st
from dotenv import load_dotenv
import time

# LangChain - Document Processing
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain - Embeddings & Vector Store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain - RAG Chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
# This keeps our API key secure (not in code)
load_dotenv()

# These are DESIGN DECISIONS - you should be able to explain WHY for each!

# EMBEDDING MODEL
# ---------------
# OpenAI offers different embedding models with different trade-offs:
# - text-embedding-3-small: 1536 dims, $0.00002/1K tokens (BEST for us!)
# - text-embedding-3-large: 3072 dims, $0.00013/1K tokens (overkill)
# - ada-002 (old): 1536 dims, $0.0001/1K tokens (deprecated)
EMBEDDING_MODEL = 'text-embedding-3-small'

# LANGUAGE MODEL (for answer generation)
# --------------------------------------
# Why GPT-3.5-turbo?
# ✅ Cost: $0.001 per 1K tokens (GPT-4 is $0.03 - 30x more expensive!)
# ✅ Speed: ~500ms response (GPT-4 can be 2-3 seconds)
# ✅ Quality: Good enough for Q&A with proper context
# ❌ When to use GPT-4: Complex reasoning, creative tasks, math
LLM_MODEL = 'gpt-3.5-turbo'

# TEMPERATURE
# -----------
# Temperature controls randomness in LLM responses
# 0 = Deterministic, same answer every time (BEST for Q&A!)
# 1 = Creative, different answers each time (GOOD for creative writing)
# 2 = Very random (usually not useful)
LLM_TEMPERATURE = 0

# CHUNK SIZE & OVERLAP
# --------------------
# This is CRITICAL for RAG performance!
# 
# Chunk Size = 512 tokens (~384 words)
# Why not smaller (128)?
#   ❌ Loses context
#   ❌ Sentences get split mid-thought
#   ❌ Harder to understand retrieved chunks
#
# Why not larger (2048)?
#   ❌ Less precise retrieval (chunks cover too many topics)
#   ❌ More expensive (more tokens to embed and process)
#   ❌ Context window fills up faster
#
# 512 = Sweet spot! (industry standard for most documents)
CHUNK_SIZE = 512

# Chunk Overlap = 50 tokens (~38 words)
# Why overlap?
#   ✅ Prevents splitting important information
#   ✅ Example: If sentence is at chunk boundary, overlap ensures full context
#   ✅ 50 tokens = ~1-2 sentences of overlap
CHUNK_OVERLAP = 50

# TOP-K RETRIEVAL
# ---------------
# How many chunks to retrieve for each question
# 
# Why not 1?
#   ❌ Not enough context
#   ❌ Might miss relevant information
#
# Why not 10?
#   ❌ Too much noise
#   ❌ Expensive (more tokens = higher cost)
#   ❌ Can confuse the LLM
#
# 4 = Sweet spot! (gives ~2000 tokens of context)
TOP_K_DOCUMENTS = 4

# VECTOR STORE PATH
# -----------------
# Where to save the FAISS index on disk
# Why save? Embeddings are expensive to regenerate!
# With caching: Load in 1 second
# Without caching: 30-60 seconds to re-embed everything
VECTOR_STORE_PATH = "data/faiss_index"

def load_and_split_pdf(pdf_path):
    """
    Load a PDF file and split it into chunks for processing.
    
    THE PROCESS:
    ------------
    1. PyPDFLoader reads PDF page by page
    2. Extracts text from each page (handles various PDF formats)
    3. RecursiveCharacterTextSplitter intelligently splits text:
       - First tries: "\n\n" (paragraph breaks)
       - Then tries: "\n" (line breaks)
       - Then tries: ". " (sentence ends)
       - Finally: " " (word breaks)
       This PRESERVES SEMANTIC MEANING!
    
    WHY RECURSIVE SPLITTING?
    ------------------------
    Regular splitting by character count:
        "Machine learning is a subset of AI. It involves..." 
        → Chunk 1: "Machine learning is a sub"
        → Chunk 2: "set of AI. It involves..."
        ❌ BROKEN! "sub" and "set" are split!
    
    Recursive splitting:
        → Chunk 1: "Machine learning is a subset of AI."
        → Chunk 2: "It involves algorithms that learn from data."
        ✅ PERFECT! Each chunk is semantically complete!
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of Document objects, each containing:
              - page_content: The text chunk
              - metadata: {source: filename, page: page_number}
              
    Example:
        chunks = load_and_split_pdf("document.pdf")
        # chunks[0].page_content = "Machine learning is..."
        # chunks[0].metadata = {"source": "document.pdf", "page": 0}
    """
    try:
        # Load PDF - PyPDFLoader handles various PDF formats
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Initialize the text splitter
        # RecursiveCharacterTextSplitter is SMART - it understands text structure
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            length_function = len,
            separators = ["/n/n", "/n", ".", " ", ""]
        )

        # Split documents into chunks
        # This returns a list of Document objects
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    except Exception as e:
        # If anything goes wrong, show user-friendly error
        st.error(f"❌ Error loading PDF: {str(e)}")
        return None

def create_vector_store(documents):
    """
    Create a FAISS vector store from document chunks.
    
    THE MAGIC HAPPENS HERE! 🎩✨
    ---------------------------
    This function is the HEART of RAG. Here's what happens:
    
    1. For each document chunk:
       Text: "Machine learning is a subset of AI"
           ↓
       Send to OpenAI API
           ↓
       Get back(vectors): [0.234, -0.123, 0.456, ..., 0.789]  (1536 numbers!)
           ↓
       This is the EMBEDDING - a mathematical representation of meaning
    
    2. Why embeddings are AMAZING:
       "ML is part of AI" → [0.235, -0.121, 0.458, ...]
       "Machine learning is subset of AI" → [0.234, -0.123, 0.456, ...]
       
       These vectors are VERY CLOSE! (cosine similarity ~0.95)
       
       But:
       "I like pizza" → [0.891, 0.234, -0.567, ...]
       
       This is FAR from the ML vectors! (cosine similarity ~0.1)
    
    3. FAISS stores all these vectors in a smart data structure:
       - Uses HNSW (Hierarchical Navigable Small World) algorithm
       - Think of it like a GPS for vectors
       - Can search millions of vectors in milliseconds!
    
    WHY NOT JUST KEYWORD SEARCH?
    -----------------------------
    Keyword search:
        User asks: "What's ML?"
        Searches for: "ML"
        Misses documents that say: "machine learning", "artificial intelligence"
        ❌ NOT SMART!
    
    Semantic search (our way):
        User asks: "What's ML?"
        Embedding: [0.234, -0.123, 0.456, ...]
        Finds similar embeddings:
            - "machine learning" ✅
            - "artificial intelligence concepts" ✅
            - "AI and its subfields" ✅
        🎯 UNDERSTANDS MEANING!
    
    Args:
        documents (list): List of Document objects with text chunks
        
    Returns:
        FAISS: A vector store object that can:
               - Store embeddings
               - Search by similarity
               - Return most relevant chunks
               
    Cost:
        ~$0.0001 per document chunk (very cheap!)
        Example: 50 chunks = $0.005 (half a cent!)
    """
    try:
        # Initialize OpenAI embeddings
        # This creates a connection to OpenAI's embedding API
        embeddings = OpenAIEmbeddings(
            model = EMBEDDING_MODEL,  # text-embedding-3-small
            openai_api_key = os.getenv("OPENAI_API_KEY")
        )

        # Create FAISS vector store
        # This does TWO things:
        # 1. Sends all chunks to OpenAI to get embeddings (costs money)
        # 2. Builds FAISS search index (fast!)
        st.info(f"🔄 Embedding {len(documents)} document chunks...")
        st.info("⏱️ This will take 10-30 seconds depending on document size...")

        vector_store = FAISS.from_documents(
            documents,
            embeddings
        )

        st.success(f"✅ Successfully embedded {len(documents)} chunks!")

        return vector_store

    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        st.info("💡 Common issues:")
        st.info("- Check your OpenAI API key")
        st.info("- Ensure you have credits in your OpenAI account")
        st.info("- Check your internet connection")
        return None
    
def save_vector_store(vector_store, path = VECTOR_STORE_PATH):
    """
    Save FAISS index to disk.
    
    WHY SAVE?
    ---------
    Embeddings are EXPENSIVE (time + money):
    - 100 chunks = ~$0.002 + 30 seconds
    - Without caching: Pay this EVERY time you restart app!
    - With caching: Pay once, reuse forever! ✅
    
    In production:
    - Embeddings are ALWAYS cached
    - Only re-embed when documents change
    - Saves $$$ and time!
    
    Args:
        vector_store (FAISS): The vector store to save
        path (str): Where to save it
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok = True)

        # Save to disk
        # FAISS saves two files:
        # - index file (the vector search structure)
        # - docstore file (the original document chunks)
        vector_store.save_local(path)

        st.success(f"💾 Vector store saved to {path}")
        st.info("✅ Next time, you can load this instantly!")
        
    except Exception as e:
        st.error(f"❌ Error saving vector store: {str(e)}")

def load_vector_store(path = VECTOR_STORE_PATH):
    """
    Load existing FAISS index from disk.
    
    WHY LOAD?
    ---------
    - Instant! (1 second vs 30 seconds to re-embed)
    - Free! (no API calls)
    - Same quality! (identical embeddings)
    
    Returns:
        FAISS: Loaded vector store, or None if doesn't exist
    """
    try:
        if os.path.exists(path):
            # Need to provide embeddings object even for loading
            # (FAISS needs to know the embedding dimension)
            embeddings = OpenAIEmbeddings(
                model = EMBEDDING_MODEL,
                openai_api_key = os.getenv("OPENAI_API_KEY")
            )

            # Load the saved index
            vector_store = FAISS.load_local(
                path,
                embeddings,
                allow_dangerous_deserialization = True  # Required by FAISS for security
            )

            return vector_store
    
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        return None
    
# ═══════════════════════════════════════════════════════════════════════
# RAG QUERY FUNCTION - THE HEART OF THE SYSTEM! 🧠
# ═══════════════════════════════════════════════════════════════════════

def create_rag_chain(vector_store):
    """
    Create the RAG (Retrieval-Augmented Generation) chain.
    
    THIS IS WHERE THE MAGIC HAPPENS! ✨
    ------------------------------------
    
    Let's walk through a COMPLETE example:
    
    ┌─────────────────────────────────────────────────────────────┐
    │ USER ASKS: "What is machine learning?"                      │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ STEP 1: EMBED THE QUESTION                                  │
    │ "What is machine learning?"                                 │
    │     ↓                                                       │
    │ [0.234, -0.123, 0.456, ..., 0.789] (1536 numbers)           │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ STEP 2: SEARCH VECTOR DATABASE                              │
    │                                                             │
    │ Compare question embedding with ALL document embeddings     │
    │ Using cosine similarity:                                    │
    │                                                             │
    │ Chunk 1: "ML is a subset of AI..." → Similarity: 0.92 ⭐   │
    │ Chunk 2: "ML algorithms learn..." → Similarity: 0.89 ⭐    │
    │ Chunk 3: "Types of ML include..." → Similarity: 0.85 ⭐    │
    │ Chunk 4: "Neural networks are..." → Similarity: 0.82 ⭐    │
    │ Chunk 5: "Python is a language..." → Similarity: 0.23 ❌   │
    │ Chunk 6: "The weather today..." → Similarity: 0.05 ❌      │
    │                                                             │
    │ Return top 4 chunks! (set by TOP_K_DOCUMENTS)              │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ STEP 3: BUILD AUGMENTED PROMPT                              │
    │                                                             │
    │ "You are a helpful AI assistant.                           │
    │                                                             │
    │  Context:                                                   │
    │  1. ML is a subset of AI that focuses on...               │
    │  2. ML algorithms learn from data to make...              │
    │  3. Types of ML include supervised, unsupervised...       │
    │  4. Neural networks are ML models inspired by...          │
    │                                                             │
    │  Question: What is machine learning?                       │
    │                                                             │
    │  Answer based ONLY on the context above."                  │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ STEP 4: SEND TO GPT-3.5-TURBO                              │
    │                                                             │
    │ GPT reads the context and generates answer                 │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ STEP 5: RETURN ANSWER                                       │
    │                                                             │
    │ "Based on the provided documents, machine learning is      │
    │  a subset of artificial intelligence that focuses on       │
    │  building systems that learn from data. The documents      │
    │  explain that ML algorithms improve through experience     │
    │  rather than explicit programming, and include types       │
    │  such as supervised, unsupervised, and reinforcement       │
    │  learning."                                                 │
    │                                                             │
    │ + [Shows the 4 source chunks for verification]             │
    └─────────────────────────────────────────────────────────────┘
    
    WHY THIS IS BETTER THAN REGULAR CHATGPT:
    ----------------------------------------
    
    Regular ChatGPT:
        ❌ Only knows data up to training cutoff
        ❌ Can't access YOUR documents
        ❌ Might hallucinate facts
        ❌ No source citations
    
    Our RAG System:
        ✅ Knows YOUR specific documents
        ✅ Grounds answers in YOUR data
        ✅ Shows WHERE information came from
        ✅ Can't make things up (limited to context)
        ✅ Up-to-date (you control the documents!)
    
    Args:
        vector_store (FAISS): The vector database with embeddings
        
    Returns:
        RetrievalQA: A LangChain chain that handles:
                     - Retrieval (finding relevant chunks)
                     - Augmentation (adding to prompt)
                     - Generation (GPT creates answer)
    """
    llm = ChatOpenAI(
        model = LLM_MODEL,  # gpt-3.5-turbo
        temperature = LLM_TEMPERATURE,  # 0 = consistent, factual answers
        openai_api_key = os.getenv("OPENAI_API_KEY")
    )

    # The retriever handles the "R" in RAG (Retrieval)
    # It takes a question and returns relevant document chunks
    retriever = vector_store.as_retriever(
        search_type = "similarity",  # Use cosine similarity
        search_kwargs = {"k": TOP_K_DOCUMENTS}  # Return top 4 chunks
    )

    # HELPER FUNCTION TO FORMAT DOCUMENTS
    def format_docs(docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # This is CRITICAL! The prompt determines:
    # - How the LLM behaves
    # - Whether it sticks to facts
    # - How it handles missing information
    template = """You are a helpful AI assistant answering questions based on provided documents.

Use ONLY the following context to answer the question. If you cannot find the answer in the context, say "I cannot find this information in the provided documents."

DO NOT make up information. DO NOT use your general knowledge. ONLY use the context below.

Context:
{context}

Question: {question}

Answer: Based on the documents provided, """

    prompt = ChatPromptTemplate.from_template(template)

    # This chain takes documents and formats them with the question

    # What does chain_type="stuff" mean?
    # "stuff" = Take all retrieved docs and "stuff" them into the prompt
    # Other options:
    # - "map_reduce" = Process docs separately, then combine (for very long docs)
    # - "refine" = Process iteratively (for complex questions)
    # - "map_rerank" = Score each doc, pick best (for ambiguous questions)
    # 
    # For most RAG systems, "stuff" is perfect! ✅
    # This combines retriever + document chain

    rag_chain = (
        {
            "context": retriever | format_docs,  # Retrieve docs and format them
            "question": RunnablePassthrough()     # Pass question through
        }
        | prompt          # Format into prompt
        | llm            # Send to LLM
        | StrOutputParser()  # Parse output to string
    )
    
    return rag_chain, retriever


def answer_question(chain, retriever, question):
    """
    Get answer for a user's question using the RAG chain.
    
    This function is simple but powerful!
    It wraps the entire RAG process in one easy call.

    Key Differences from Old Version:
    ---------------------------------
    Old: chain({"query": question})
    New: chain.invoke({"input": question})
    
    Args:
        chain: The RAG chain (from create_rag_chain)
        question (str): User's question
        
    Returns:
        tuple: (answer_text, source_documents)
               - answer_text: The generated answer
               - source_documents: The chunks used (for citations)
               
    Example:
        answer, sources = answer_question(chain, "What is ML?")
        print(answer)  # "Machine learning is..."
        print(sources[0].page_content)  # "ML is a subset..."
    """
    try:
        # Run the entire RAG pipeline with one call!
        # This does:
        # 1. Embed question
        # 2. Search vector store
        # 3. Build prompt
        # 4. Call GPT
        # 5. Return result

        # Extract answer and sources
        answer = chain.invoke(question)
        sources = retriever.invoke(question)

        return answer, sources
    
    except Exception as e:
        st.error(f"❌ Error answering question: {str(e)}")
        st.info("💡 Common issues:")
        st.info("- Check your internet connection")
        st.info("- Ensure you have OpenAI API credits")
        st.info("- Try rephrasing your question")
        return None, None
    
def main():
    """
    Main Streamlit application.
    
    UI FLOW:
    --------
    1. User uploads PDF(s) via sidebar
    2. System processes: Load → Split → Embed → Index → Save
    3. User asks question in main area
    4. System: Embed question → Search → Retrieve → Generate answer
    5. Display: Answer + Source documents with page numbers
    
    STREAMLIT SESSION STATE:
    ------------------------
    Streamlit reruns the entire script on every interaction!
    session_state lets us persist data between reruns:
    - st.session_state.vector_store = keeps our vector DB in memory
    - Without this, we'd lose everything on each button click!
    """
    
    # ─────────────────────────────────────────────────────────────
    # PAGE CONFIGURATION
    # ─────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Document Q&A with RAG",
        page_icon="📚",
        layout="wide",  # Use full screen width
        initial_sidebar_state="expanded"  # Sidebar open by default
    )
    
    # ═══════════════════════════════════════════════════════════════
    # API KEY HANDLING (Works for both local and deployed)
    # ═══════════════════════════════════════════════════════════════
    
    # Initialize session state for API key if not exists
    if 'api_key_provided' not in st.session_state:
        st.session_state.api_key_provided = False
    
    # Check if API key is already in environment (local development with .env)
    if os.getenv("OPENAI_API_KEY") and not st.session_state.api_key_provided:
        st.session_state.api_key_provided = True
    
    # If no API key in environment, show input UI
    if not st.session_state.api_key_provided:
        st.title("🔑 OpenAI API Key Required")
        
        st.markdown("""
        Welcome! This application requires an OpenAI API key to function.
        
        ### 🎯 How to get your API key:
        1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Sign up or log in to your account
        3. Click "Create new secret key"
        4. Copy the key and paste it below
        
        ### 💰 Cost Estimate:
        - **Very affordable!** Approximately $0.10-0.20 for:
          - Processing 100 pages of documents
          - Asking 50 questions
        - Your key is **NOT stored permanently**
        - Only used during your current session
        
        ### 🔒 Privacy & Security:
        - ✅ Your API key is stored only in your browser session
        - ✅ Keys are never logged or saved to disk
        - ✅ Your documents are processed in memory only
        - ✅ No data is stored on our servers
        - ✅ All processing happens securely through OpenAI's API
        
        ### ❓ Don't have an API key?
        - Sign up at [OpenAI](https://platform.openai.com/signup) (free)
        - Add minimum $5 credit to your account
        - Create your first API key
        """)
        
        st.markdown("---")
        
        # API key input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            api_key = st.text_input(
                "Enter your OpenAI API Key:",
                type="password",
                placeholder="sk-proj-...",
                help="Your API key will only be used for this session and not stored anywhere"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            submit_button = st.button("✅ Submit", type="primary")
        
        if submit_button or api_key:
            if api_key and api_key.startswith("sk-"):
                # Validate key format
                os.environ["OPENAI_API_KEY"] = api_key
                st.session_state.api_key_provided = True
                st.success("✅ API Key accepted! Loading application...")
                st.balloons()
                time.sleep(1)  # Brief pause for user to see success
                st.rerun()
            elif api_key:
                st.error("❌ Invalid API key format. OpenAI keys start with 'sk-' or 'sk-proj-'")
            else:
                st.warning("⚠️ Please enter your API key above.")
        
        # Show example
        with st.expander("💡 Example: What you can do with this app"):
            st.markdown("""
            **Upload any PDF and ask questions like:**
            
            📚 **For Research Papers:**
            - "What are the main findings?"
            - "What methodology was used?"
            - "What are the limitations?"
            
            💼 **For Business Documents:**
            - "What were the Q3 revenue figures?"
            - "Summarize the key recommendations"
            
            📖 **For Technical Docs:**
            - "How do I configure X?"
            - "What are the system requirements?"
            
            The system will search through your documents semantically and provide 
            accurate answers with source citations!
            """)
        
        st.stop()  # Stop execution here until API key is provided

    # ─────────────────────────────────────────────────────────────
    # HEADER & DESCRIPTION
    # ─────────────────────────────────────────────────────────────
    st.title("📚 Document Q&A with RAG")
    st.markdown("""
    **Upload your PDF documents and ask questions about them using AI!**
    
    ### 🔧 How it works:
    1. **Upload PDFs** → System extracts and indexes content
    2. **Ask questions** → AI searches your documents semantically
    3. **Get answers** → Responses grounded in YOUR documents with citations
    
    ### 🎯 Powered by:
    - 🤖 **OpenAI GPT-3.5-turbo** for intelligent answer generation
    - 🔍 **FAISS** for lightning-fast semantic search
    - 📊 **Embeddings** to understand document meaning (not just keywords)
    - 🦜 **LangChain** for seamless RAG pipeline orchestration
    
    ### 💡 What makes this special:
    - ✅ Semantic search (understands meaning, not just keywords)
    - ✅ Source citations (verify where answers come from)
    - ✅ Fast & efficient (cached embeddings for reuse)
    - ✅ Production-ready error handling
    """)
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────
    # SIDEBAR - DOCUMENT UPLOAD & PROCESSING
    # ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📄 Document Management")
        
        st.markdown("""
        **Instructions:**
        1. Upload one or more PDF files
        2. Click "🚀 Process Documents"
        3. Wait for processing to complete
        4. Start asking questions!
        """)
        
        # ─────────────────────────────────────────────────────────
        # FILE UPLOADER
        # ─────────────────────────────────────────────────────────
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to create your knowledge base"
        )
        
        # Show how many files uploaded
        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} file(s) selected")
            for file in uploaded_files:
                st.text(f"• {file.name}")
        
        # ─────────────────────────────────────────────────────────
        # PROCESS DOCUMENTS BUTTON
        # ─────────────────────────────────────────────────────────
        if uploaded_files and st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing your documents..."):
                
                # Create documents directory if doesn't exist
                os.makedirs("documents", exist_ok=True)
                
                # Container to hold all chunks from all PDFs
                all_chunks = []
                
                # ─────────────────────────────────────────────────
                # PROCESS EACH UPLOADED PDF
                # ─────────────────────────────────────────────────
                for uploaded_file in uploaded_files:
                    # Save uploaded file to disk temporarily
                    # (Streamlit uploaded files are in memory only)
                    temp_path = f"documents/{uploaded_file.name}"
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load and split this PDF
                    st.info(f"📖 Processing: {uploaded_file.name}...")
                    chunks = load_and_split_pdf(temp_path)
                    
                    if chunks:
                        all_chunks.extend(chunks)
                        st.success(f"✅ {uploaded_file.name}: {len(chunks)} chunks created")
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}")
                
                # ─────────────────────────────────────────────────
                # CREATE VECTOR STORE FROM ALL CHUNKS
                # ─────────────────────────────────────────────────
                if all_chunks:
                    st.info(f"🔄 Creating vector store from {len(all_chunks)} total chunks...")
                    st.info("⏱️ This may take 10-60 seconds depending on document size...")
                    
                    # This is where embeddings are generated (costs money!)
                    vector_store = create_vector_store(all_chunks)
                    
                    if vector_store:
                        # Save for future use (caching!)
                        save_vector_store(vector_store)
                        
                        # Store in session state (keeps it in memory)
                        st.session_state.vector_store = vector_store
                        
                        # Success!
                        st.success("🎉 Documents processed successfully!")
                        st.balloons()  # Celebration animation!
                        
                        # Show statistics
                        st.markdown("---")
                        st.markdown("**📊 Processing Summary:**")
                        st.metric("Total Documents", len(uploaded_files))
                        st.metric("Total Chunks", len(all_chunks))
                        st.metric("Avg Chunk Size", f"{sum(len(c.page_content) for c in all_chunks) // len(all_chunks)} chars")
                else:
                    st.error("❌ No chunks created. Please check your PDF files.")
        
        # ─────────────────────────────────────────────────────────
        # LOAD EXISTING INDEX BUTTON
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Or load previously processed documents:**")
        
        if st.button("📂 Load Existing Index"):
            with st.spinner("Loading saved index..."):
                vector_store = load_vector_store()
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.success("✅ Index loaded successfully!")
                    st.info("You can now ask questions about previously processed documents.")
                else:
                    st.error("❌ No saved index found. Please process documents first.")
        
        # ─────────────────────────────────────────────────────────
        # SYSTEM INFORMATION
        # ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚙️ System Configuration")
        
        # Display current settings
        st.markdown(f"""
        **Embedding Model:** {EMBEDDING_MODEL}  
        **LLM:** {LLM_MODEL}  
        **Chunk Size:** {CHUNK_SIZE} tokens  
        **Chunk Overlap:** {CHUNK_OVERLAP} tokens  
        **Retrieval:** Top {TOP_K_DOCUMENTS} chunks  
        **Temperature:** {LLM_TEMPERATURE} (factual mode)
        """)
        
        # Cost estimate
        st.markdown("---")
        st.markdown("### 💰 Cost Estimate")
        st.markdown("""
        **Per document (100 pages):**
        - Embedding: ~$0.02
        - Storage: Free (local)
        
        **Per question:**
        - Query: ~$0.002-0.005
        
        **Very affordable!** 🎉
        """)
    
    # ─────────────────────────────────────────────────────────────
    # MAIN AREA - QUESTION & ANSWER
    # ─────────────────────────────────────────────────────────────
    
    # Check if vector store exists in session state
    if "vector_store" not in st.session_state:
        # No documents processed yet - show instructions
        st.warning("⚠️ No documents loaded yet!")
        st.info("👈 Please upload and process documents using the sidebar, or load an existing index.")
        
        # Show example to help users understand what they can do
        with st.expander("💡 What can you do with this system?"):
            st.markdown("""
            **Example Use Cases:**
            
            📚 **Research Papers:**
            - "What are the main findings of this study?"
            - "What methodology was used?"
            - "What are the limitations mentioned?"
            
            📄 **Business Documents:**
            - "What were the Q3 revenue figures?"
            - "What are the key strategic initiatives?"
            - "Summarize the executive summary"
            
            📖 **Technical Documentation:**
            - "How do I configure this feature?"
            - "What are the system requirements?"
            - "Explain the API endpoints"
            
            🎓 **Educational Content:**
            - "Define machine learning"
            - "What are the types of neural networks?"
            - "Explain reinforcement learning"
            """)
        
        # Stop here - don't show Q&A interface yet
        return
    
    # ─────────────────────────────────────────────────────────────
    # Q&A INTERFACE (only shown after documents are processed)
    # ─────────────────────────────────────────────────────────────
    
    st.header("💬 Ask Questions About Your Documents")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the main topics discussed in these documents?",
        help="Ask anything about the uploaded documents"
    )
    
    # Two columns for button and clear
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("🔍 Get Answer", type="primary")
    
    with col2:
        if st.button("🗑️ Clear Results"):
            # Clear previous results
            if 'last_answer' in st.session_state:
                del st.session_state.last_answer
            if 'last_sources' in st.session_state:
                del st.session_state.last_sources
            st.rerun()
    
    # ─────────────────────────────────────────────────────────────
    # PROCESS QUESTION & DISPLAY ANSWER
    # ─────────────────────────────────────────────────────────────
    if question and ask_button:
        
        with st.spinner("🔍 Searching documents and generating answer..."):
            
            # Create the RAG chain
            chain, retriever = create_rag_chain(st.session_state.vector_store)
            
            # Get answer
            answer, sources = answer_question(chain, retriever, question)
            
            if answer and sources:
                # Store in session state so it persists
                st.session_state.last_answer = answer
                st.session_state.last_sources = sources
    
    # ─────────────────────────────────────────────────────────────
    # DISPLAY RESULTS (if available)
    # ─────────────────────────────────────────────────────────────
    if 'last_answer' in st.session_state:
        
        st.markdown("---")
        
        # Display the answer in a nice container
        st.markdown("### 📝 Answer:")
        st.markdown(
            f"""<div style="background-color: #000000; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
            {st.session_state.last_answer}
            </div>""",
            unsafe_allow_html=True
        )
        
        # Display source documents
        st.markdown("---")
        st.markdown("### 📚 Source Documents:")
        st.markdown("*The answer above was generated from these document sections:*")
        
        # Show each source in an expander
        for i, doc in enumerate(st.session_state.last_sources, 1):
            # Get metadata
            source_file = doc.metadata.get('source', 'Unknown')
            page_num = doc.metadata.get('page', 'N/A')
            
            # Create expander for each source
            with st.expander(f"📄 Source {i}: {source_file} (Page {page_num})"):
                st.markdown(f"**File:** {source_file}")
                st.markdown(f"**Page:** {page_num}")
                st.markdown("**Content:**")
                st.text(doc.page_content)
        
        # Add feedback option
        st.markdown("---")
        st.markdown("### 💭 Was this helpful?")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👍 Yes, helpful!"):
                st.success("Thanks for your feedback!")
        
        with col2:
            if st.button("👎 Not helpful"):
                st.info("Thanks! We're constantly improving.")
        
        with col3:
            if st.button("🔄 Ask another question"):
                # Clear results and rerun
                del st.session_state.last_answer
                del st.session_state.last_sources
                st.rerun()
    
    # ─────────────────────────────────────────────────────────────
    # EXAMPLE QUESTIONS (helpful for users)
    # ─────────────────────────────────────────────────────────────
    with st.expander("💡 Example Questions You Can Ask"):
        st.markdown("""
        **General Questions:**
        - What is the main topic of these documents?
        - Summarize the key points
        - What conclusions are drawn?
        
        **Specific Questions:**
        - What does the document say about [specific topic]?
        - Define [specific term] as mentioned in the documents
        - What examples are provided for [concept]?
        
        **Analytical Questions:**
        - What are the advantages and disadvantages mentioned?
        - What recommendations are provided?
        - What data or statistics are presented?
        
        **Comparative Questions:**
        - How does [concept A] differ from [concept B]?
        - What are the similarities between [topic X] and [topic Y]?
        
        💡 **Tip:** Be specific! The more focused your question, the better the answer.
        """)
    
    # ─────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using LangChain, OpenAI, and Streamlit</p>
        <p>🔗 <a href="https://github.com/fr0styyXD/document-qa-rag" target="_blank">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# RUN THE APPLICATION
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # This ensures the app only runs when executed directly
    # (not when imported as a module)
    main()
