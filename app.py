import time
import fitz
import numpy as np
import dill
import os
import logging
import networkx as nx
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from typing import List, Optional, Tuple
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_openai_client():
    """Get OpenAI client with proper error handling"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)

PDF_PATH = "input.pdf"
VECTOR_DB_PATH = "vector_db.pkl"
ANNOY_INDEX_PATH = "vector_index.ann"

def get_text_embedding_with_rate_limit(text_list, initial_delay=2, max_retries=10, max_delay=60):
    client = get_openai_client()
    embeddings = []
    for text in text_list:
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                if len(text) > 8192:
                    logging.warning("Text chunk exceeds the token limit. Truncating the text.")
                    text = text[:8192]
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[text]
                )
                embeddings.append(response.data[0].embedding)
                time.sleep(delay)
                break
            except Exception as e:
                retries += 1
                logging.warning(f"Embedding retry {retries}/{max_retries} after error: {e}")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
                if retries == max_retries:
                    logging.error("Max retries reached. Skipping this chunk.")
    return embeddings

def split_text_into_chunks(text: str, chunk_size: int = 2048, overlap: int = 200) -> List[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def store_embeddings_in_vector_db(pdf_path, vector_db_path, annoy_index_path, chunk_size=2048, overlap=200, num_trees=10):
    doc = fitz.open(pdf_path)
    all_embeddings = []
    all_texts = []
    for page_num in range(doc.page_count):
        text = doc.load_page(page_num).get_text()
        if text.strip():
            chunks = split_text_into_chunks(text, chunk_size, overlap)
            embeddings = get_text_embedding_with_rate_limit(chunks)
            all_embeddings.extend(embeddings)
            all_texts.extend(chunks)
    embeddings_np = np.array(all_embeddings).astype('float32')
    with open(vector_db_path, "wb") as f:
        dill.dump({'embeddings': embeddings_np, 'texts': all_texts}, f)
    if os.path.exists(annoy_index_path):
        os.remove(annoy_index_path)
    embedding_dim = embeddings_np.shape[1]
    annoy_index = AnnoyIndex(embedding_dim, 'angular')
    for i, embedding in enumerate(embeddings_np):
        annoy_index.add_item(i, embedding)
    annoy_index.build(num_trees)
    annoy_index.save(annoy_index_path)

def initialize_vector_database():
    """Initialize vector database if it doesn't exist"""
    if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(ANNOY_INDEX_PATH):
        if not os.path.exists(PDF_PATH):
            st.error(f"PDF file not found: {PDF_PATH}")
            st.stop()
        
        # Check if OpenAI API key is available
        try:
            get_openai_client()
        except ValueError as e:
            st.error(f"OpenAI API Key Error: {e}")
            st.info("Please set your OPENAI_API_KEY environment variable and restart the application.")
            st.stop()
        
        with st.spinner("Processing PDF and creating vector database... This may take a few minutes."):
            store_embeddings_in_vector_db(PDF_PATH, VECTOR_DB_PATH, ANNOY_INDEX_PATH)
        st.success("Vector database created successfully!")
        st.rerun()

class MistralRAGChatbot:
    def __init__(self, vector_db_path: str, annoy_index_path: str):
        with open(vector_db_path, "rb") as f:
            data = dill.load(f)
        self.embeddings = np.array(data['embeddings'], dtype='float32')
        self.texts = data['texts']
        self.annoy_index = AnnoyIndex(self.embeddings.shape[1], 'angular')
        self.annoy_index.load(annoy_index_path)
        self.bm25 = BM25Okapi([text.split() for text in self.texts])

    def get_text_embedding(self, text: str) -> np.ndarray:
        try:
            client = get_openai_client()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=[text]
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error fetching embedding: {e}")
            return np.zeros((1536,), dtype=np.float32)

    def retrieve_documents(self, query: str, embedding: np.ndarray, top_k=10):
        indices, distances = self.annoy_index.get_nns_by_vector(embedding, top_k, include_distances=True)
        bm25_scores = self.bm25.get_scores(query.split())
        combined_docs = []
        for idx in indices:
            combined_docs.append({
                'text': self.texts[idx],
                'method': 'hybrid',
                'score': float(bm25_scores[idx]),
                'index': idx
            })
        return combined_docs

    def rerank_documents(self, query: str, docs: List[dict]) -> List[dict]:
        query_embedding = self.get_text_embedding(query)
        vector_scores = {doc['index']: doc['score'] for doc in docs}
        sim_graph = nx.Graph()
        sim_matrix = cosine_similarity(self.embeddings)
        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                if sim_matrix[i, j] > 0.5:
                    sim_graph.add_edge(i, j, weight=sim_matrix[i, j])
        pagerank_scores = np.array(list(nx.pagerank(sim_graph, weight='weight').values()))
        for doc in docs:
            idx = doc['index']
            doc['score'] = 0.7 * vector_scores.get(idx, 0) + 0.3 * pagerank_scores[idx]
        return sorted(docs, key=lambda x: x['score'], reverse=True)[:5]

    def build_prompt(self, context: str, query: str, style: str) -> str:
        styles = {
            "detailed": "Provide a detailed answer.",
            "concise": "Provide a concise answer.",
            "creative": "Be creative in your response.",
            "technical": "Provide a technically sound answer."
        }
        instruction = styles.get(style.lower(), styles["detailed"])
        return f"""You are a helpful assistant.\nContext:\n{context}\nQuestion:\n{query}\nInstruction:\n{instruction}"""

    def generate_response(self, query: str, style: str) -> str:
        query_embedding = self.get_text_embedding(query)
        docs = self.retrieve_documents(query, query_embedding)
        reranked_docs = self.rerank_documents(query, docs)
        context = "\n\n".join([doc['text'] for doc in reranked_docs])
        prompt = self.build_prompt(context, query, style)
        try:
            client = get_openai_client()
            response = ""
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            return response
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response."

def main():
    st.set_page_config(
        page_title="ChanceRAG",
        page_icon="🤖",
        layout="wide"
    )
    
    # Check for API key first
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("🔑 OpenAI API Key Required")
        st.markdown("""
        **Please set your OpenAI API key in the .env file:**
        
        Create a `.env` file in the project directory and add:
        ```
        OPENAI_API_KEY=your-api-key-here
        ```
        
        Or set it as an environment variable:
        
        **Windows (PowerShell):**
        ```powershell
        $env:OPENAI_API_KEY="your-api-key-here"
        ```
        
        **Windows (Command Prompt):**
        ```cmd
        set OPENAI_API_KEY=your-api-key-here
        ```
        
        **Unix/Linux/Mac:**
        ```bash
        export OPENAI_API_KEY="your-api-key-here"
        ```
        
        After setting the API key, restart the application.
        """)
        st.stop()
    
    # Initialize vector database if needed
    initialize_vector_database()
    
    # Display logo
    st.image("images/chanceRAG_logo.jpg", width=400)
    
    st.title("ChanceRAG - Document Question Answering")
    st.markdown("Ask questions about the uploaded PDF document and get intelligent responses.")
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User input
        user_query = st.text_area(
            "Enter your question:",
            height=150,
            placeholder="Type your question about the document here..."
        )
    
    with col2:
        # Response style selection
        response_style = st.selectbox(
            "Response Style:",
            ["Detailed", "Concise", "Creative", "Technical"],
            index=0
        )
        
        # Submit button
        submit_button = st.button("Get Answer", type="primary", use_container_width=True)
    
    # Process query when button is clicked
    if submit_button and user_query.strip():
        with st.spinner("Generating response..."):
            try:
                bot = MistralRAGChatbot(VECTOR_DB_PATH, ANNOY_INDEX_PATH)
                response = bot.generate_response(user_query, response_style)
                
                st.subheader("ChanceRAG Response:")
                st.write(response)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please make sure the PDF has been processed and try again.")
    
    elif submit_button and not user_query.strip():
        st.warning("Please enter a question before submitting.")

if __name__ == "__main__":
    main()
