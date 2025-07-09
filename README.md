# ChanceRAG - Streamlit Application

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that answers questions about PDF documents using advanced AI techniques.

## Features

- **Hybrid Retrieval**: Combines vector similarity search with BM25 scoring
- **Advanced Reranking**: Uses PageRank algorithm on similarity graphs
- **Multiple Response Styles**: Detailed, Concise, Creative, and Technical responses
- **Real-time Processing**: Streamlit interface for interactive querying
- **OpenAI Integration**: Uses GPT-4 for response generation

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

1. Place your PDF file as `input.pdf` in the project directory
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Or use the helper script:
   ```bash
   python run_app.py
   ```
3. Open your browser and navigate to the provided local URL
4. Enter your questions and select your preferred response style

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
