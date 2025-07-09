# ChanceRAG - Streamlit Application

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that answers questions about PDF documents using advanced AI techniques.

## Features

- **Hybrid Retrieval**: Combines vector similarity search with BM25 scoring
- **Advanced Reranking**: Uses PageRank algorithm on similarity graphs
- **Multiple Response Styles**: Detailed, Concise, Creative, and Technical responses
- **Real-time Processing**: Streamlit interface for interactive querying
- **OpenAI Integration**: Uses GPT-4 for response generation

## Installation & Setup

### Quick Start (Automated)

The easiest way to get started is using the automated setup scripts:

**Windows (PowerShell - Recommended):**
```powershell
.\start_app.ps1
```

**Windows (Command Prompt):**
```cmd
start_app.bat
```

**Any Platform:**
```bash
python run_app.py
```

These scripts will automatically:
1. Create a virtual environment
2. Install all dependencies
3. Check for OpenAI API key
4. Launch the Streamlit application

### Manual Setup

If you prefer to set up manually:

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **Windows**: `venv\Scripts\activate`
   - **Unix/Linux/Mac**: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. **Set up your OpenAI API key** (Choose one method):
   
   **Method 1: .env file (Recommended)**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your API key
   OPENAI_API_KEY=your-actual-api-key-here
   ```
   
   **Method 2: Environment variables**
   - **Windows (PowerShell)**: `$env:OPENAI_API_KEY="your-api-key-here"`
   - **Windows (CMD)**: `set OPENAI_API_KEY=your-api-key-here`
   - **Unix/Linux/Mac**: `export OPENAI_API_KEY="your-api-key-here"`

## Usage

### Quick Start
1. Place your PDF file as `input.pdf` in the project directory
2. Run the automated setup script:
   - **Windows (PowerShell)**: `.\start_app.ps1`
   - **Windows (CMD)**: `start_app.bat`
   - **Any Platform**: `python run_app.py`
3. The application will automatically open in your default browser
4. Enter your questions and select your preferred response style

### Manual Run
If you've already set up the environment manually:
```bash
streamlit run app.py
```

### First-Time Setup
On first run, the application will:
- Process your PDF document into chunks
- Generate embeddings for each chunk
- Build the vector database and search index
- This may take a few minutes depending on PDF size

## How It Works

1. **Document Processing**: PDF is chunked into overlapping segments
2. **Embedding Generation**: Text chunks are converted to embeddings using OpenAI's text-embedding-3-small
3. **Vector Storage**: Embeddings are stored using Annoy index for fast similarity search
4. **Query Processing**: User queries are embedded and matched against document chunks
5. **Reranking**: Results are reranked using BM25 scores and PageRank on similarity graphs
6. **Response Generation**: Context is provided to GPT-4 for intelligent response generation

## Configuration

- Chunk size: 2048 tokens with 200 token overlap
- Vector dimensions: 1536 (OpenAI embedding size)
- Top-k retrieval: 10 documents
- Final context: Top 5 reranked documents
