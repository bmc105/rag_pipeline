#!/usr/bin/env python3
"""
Streamlined PDF RAG Pipeline with FAISS and KoboldCpp/llama.cpp

A clean, modular pipeline for:
- PDF extraction and chunking
- FAISS vector database creation
- Retrieval-augmented generation with streaming LLM responses

Usage:
  python main.py index --pdf file1.pdf file2.pdf
  python main.py query --query "What is the main topic?"
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import fitz
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss
except ImportError:
    raise ImportError("faiss is required. Install with: pip install faiss-cpu")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    pdf_name: str
    page: int
    chunk_id: int
    text: str
    
    def to_dict(self) -> Dict:
        return {
            'pdf': self.pdf_name,
            'page': self.page,
            'chunk_id': self.chunk_id,
            'text': self.text
        }


@dataclass
class RetrievedChunk:
    """Retrieved chunk with relevance score."""
    rank: int
    score: float
    metadata: ChunkMetadata
    
    def __str__(self) -> str:
        preview = self.metadata.text[:200] + "..." if len(self.metadata.text) > 200 else self.metadata.text
        return f"Rank {self.rank} (score: {self.score:.4f}) - {self.metadata.pdf_name} p{self.metadata.page}: {preview}"


class PDFProcessor:
    """Handles PDF text extraction and chunking."""
    
    @staticmethod
    def extract_text(pdf_path: Path) -> List[str]:
        """Extract text from each page of a PDF."""
        try:
            doc = fitz.open(str(pdf_path))
            pages = [doc[i].get_text() for i in range(len(doc))]
            doc.close()
            return pages
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return []
    
    @staticmethod
    def extract_to_markdown(pdf_path: Path, output_dir: Path = Path("outputs")) -> str:
        """Convert PDF to markdown format and save to outputs directory."""
        try:
            output_dir.mkdir(exist_ok=True)
            
            doc = fitz.open(str(pdf_path))
            markdown_lines = []
            
            # Add document title
            pdf_name = pdf_path.stem
            markdown_lines.append(f"# {pdf_name}")
            markdown_lines.append("")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Add page header
                markdown_lines.append(f"## Page {page_num + 1}")
                markdown_lines.append("")
                
                # Extract text blocks with formatting
                blocks = page.get_text("dict")
                current_paragraph = []
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if not text:
                                    continue
                                    
                                # Basic formatting based on font size
                                font_size = span.get("size", 12)
                                flags = span.get("flags", 0)
                                
                                # Bold text (flag 16)
                                if flags & 16:
                                    text = f"**{text}**"
                                
                                # Italic text (flag 2)
                                if flags & 2:
                                    text = f"*{text}*"
                                
                                # Large text as headers (rough heuristic)
                                if font_size > 14:
                                    text = f"### {text}"
                                
                                line_text += text + " "
                            
                            if line_text.strip():
                                # Check if this looks like a header
                                clean_text = line_text.strip()
                                if clean_text.startswith("###") or clean_text.isupper() and len(clean_text) < 100:
                                    # Finish current paragraph
                                    if current_paragraph:
                                        markdown_lines.append(" ".join(current_paragraph))
                                        markdown_lines.append("")
                                        current_paragraph = []
                                    # Add header
                                    if not clean_text.startswith("###"):
                                        clean_text = f"### {clean_text}"
                                    markdown_lines.append(clean_text)
                                    markdown_lines.append("")
                                else:
                                    current_paragraph.append(clean_text)
                        
                        # End of block - finish paragraph
                        if current_paragraph:
                            markdown_lines.append(" ".join(current_paragraph))
                            markdown_lines.append("")
                            current_paragraph = []
                
                # Add page break
                if page_num < len(doc) - 1:
                    markdown_lines.append("---")
                    markdown_lines.append("")
            
            doc.close()
            
            # Clean up excessive blank lines
            cleaned_lines = []
            prev_empty = False
            for line in markdown_lines:
                if line.strip() == "":
                    if not prev_empty:
                        cleaned_lines.append(line)
                    prev_empty = True
                else:
                    cleaned_lines.append(line)
                    prev_empty = False
            
            markdown_content = "\n".join(cleaned_lines)
            
            # Save to file
            output_file = output_dir / f"{pdf_name}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Markdown saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to convert {pdf_path} to markdown: {e}")
            return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
            
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
            
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:  # Avoid empty chunks
                chunks.append(" ".join(chunk_words))
                
            # Stop if we've included all words
            if i + chunk_size >= len(words):
                break
                
        return chunks


class VectorStore:
    """Handles embedding generation and FAISS index operations."""
    
    def __init__(self, model_name: str = "models/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Generate normalized embeddings for texts."""
        self._load_model()
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        # Combine and normalize
        all_embeddings = np.vstack(embeddings).astype(np.float32)
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        return all_embeddings / norms
    
    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors (cosine similarity)
        index.add(embeddings)
        return index
    
    def search(self, index: faiss.Index, query: str, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search index with query."""
        self._load_model()
        
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12)
        
        scores, indices = index.search(query_embedding, top_k)
        return scores[0], indices[0]


class LLMClient:
    """Handles communication with KoboldCpp/llama.cpp server."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5001):
        self.host = host
        self.port = port
        self.endpoints = [
            f"http://{host}:{port}/api/v1/generate",  # KoboldCpp
            f"http://{host}:{port}/completion",       # llama.cpp
            f"http://{host}:{port}/v1/completions"    # OpenAI-compatible
        ]
    
    def stream_completion(self, prompt: str, max_tokens: int = 512) -> str:
        """Stream completion from LLM server."""
        payloads = [
            # KoboldCpp format
            {
                "prompt": prompt,
                "max_length": max_tokens,
                "max_context_length": 32768,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "rep_pen": 1.1,
                "stream": True,
            },
            # llama.cpp format
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "stream": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
            },
            # OpenAI format
            {
                "model": "local",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
                "temperature": 0.7,
            }
        ]
        
        for endpoint, payload in zip(self.endpoints, payloads):
            try:
                return self._try_endpoint(endpoint, payload)
            except requests.RequestException as e:
                logger.warning(f"Endpoint {endpoint} failed: {e}")
                continue
        
        logger.error(f"All endpoints failed. Ensure server is running at {self.host}:{self.port}")
        return ""
    
    def _try_endpoint(self, url: str, payload: Dict) -> str:
        """Try a specific endpoint for streaming."""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        full_text = ""
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
                
            # Handle Server-Sent Events
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str.strip() in ['[DONE]', '[END]']:
                    break
                
                token = self._extract_token(data_str)
                if token:
                    print(token, end='', flush=True)
                    full_text += token
            
            # Handle direct JSON
            elif line.startswith('{'):
                token = self._extract_token(line)
                if token:
                    print(token, end='', flush=True)
                    full_text += token
        
        print()  # Final newline
        return full_text
    
    def _extract_token(self, data_str: str) -> Optional[str]:
        """Extract token from various response formats."""
        try:
            data = json.loads(data_str)
            
            # Various response formats
            if 'token' in data:
                return data['token']
            elif 'content' in data:
                return data['content']
            elif 'choices' in data and data['choices']:
                choice = data['choices'][0]
                if 'text' in choice:
                    return choice['text']
                elif 'delta' in choice and 'content' in choice['delta']:
                    return choice['delta']['content']
                    
        except json.JSONDecodeError:
            # Handle plain text responses
            return data_str if data_str.strip() else None
            
        return None


class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self, model_name: str = "models/all-mpnet-base-v2"):
        self.vector_store = VectorStore(model_name)
        self.pdf_processor = PDFProcessor()
        
    def build_index(self, 
                   pdf_paths: List[str],
                   index_file: str = "pdf_index.faiss",
                   meta_file: str = "pdf_meta.ndjson",
                   chunk_size: int = 400,
                   overlap: int = 50) -> None:
        """Build FAISS index from PDF files."""
        
        texts = []
        metadatas = []
        
        for pdf_path_str in pdf_paths:
            pdf_path = Path(pdf_path_str)
            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                continue
                
            logger.info(f"Processing {pdf_path.name}")
            pages = self.pdf_processor.extract_text(pdf_path)
            
            for page_idx, page_text in enumerate(pages):
                chunks = self.pdf_processor.chunk_text(page_text, chunk_size, overlap)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk.strip():  # Skip empty chunks
                        texts.append(chunk)
                        metadata = ChunkMetadata(
                            pdf_name=pdf_path.name,
                            page=page_idx + 1,
                            chunk_id=chunk_idx,
                            text=chunk
                        )
                        metadatas.append(metadata)
        
        if not texts:
            raise ValueError("No text extracted from PDFs")
            
        logger.info(f"Total chunks: {len(texts)}")
        
        # Generate embeddings and build index
        embeddings = self.vector_store.embed_texts(texts)
        index = self.vector_store.create_index(embeddings)
        
        # Save index and metadata
        faiss.write_index(index, index_file)
        self._save_metadata(meta_file, metadatas)
        
        logger.info(f"Index saved to {index_file}")
        logger.info(f"Metadata saved to {meta_file}")
    
    def query(self,
             query: str,
             index_file: str = "pdf_index.faiss",
             meta_file: str = "pdf_meta.ndjson",
             top_k: int = 5,
             llm_client: Optional[LLMClient] = None,
             max_tokens: int = 512,
             custom_prompt: Optional[str] = None) -> str:
        """Query the index and optionally generate response with LLM."""
        
        # Load index and metadata
        if not Path(index_file).exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        index = faiss.read_index(index_file)
        metadatas = self._load_metadata(meta_file)
        
        # Search
        scores, indices = self.vector_store.search(index, query, top_k)
        
        # Retrieve chunks
        retrieved = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            if 0 <= idx < len(metadatas):
                chunk = RetrievedChunk(
                    rank=rank,
                    score=float(score),
                    metadata=metadatas[idx]
                )
                retrieved.append(chunk)
        
        # Display results
        logger.info("Retrieved chunks:")
        for chunk in retrieved:
            logger.info(f"  {chunk}")
        
        # Generate response if LLM client provided
        if llm_client:
            context = self._build_context(retrieved)
            prompt = self._build_prompt(query, context, custom_prompt)
            
            logger.info("Generating response...")
            logger.info("-" * 50)
            response = llm_client.stream_completion(prompt, max_tokens)
            logger.info("-" * 50)
            return response
        
        return "\n".join([chunk.metadata.text for chunk in retrieved])
    
    def _build_context(self, retrieved: List[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for chunk in retrieved:
            context_parts.append(
                f"(Source: {chunk.metadata.pdf_name}, Page {chunk.metadata.page}):\n"
                f"{chunk.metadata.text}"
            )
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, custom_prompt: Optional[str] = None) -> str:
        """Build prompt for LLM."""
        if custom_prompt:
            return custom_prompt.format(context=context, question=query)
        
        return (
            "You are a helpful technical assistant specializing in analyzing and processing specification documents. "
            "Your role is to provide accurate, detailed responses based on the specification content provided.\n\n"
            
            "INSTRUCTIONS:\n"
            "• Answer questions using ONLY the specification document content provided in the CONTEXT section\n"
            "• Be precise and technical when explaining specifications, requirements, or procedures\n"
            "• When referencing specific sections, mention the source document and page number if available\n"
            "• If asked about implementation details, provide step-by-step guidance from the specifications\n"
            "• For requirements or standards, quote the exact specification text when relevant\n"
            "• If information spans multiple sections, synthesize the content coherently\n"
            "• If the specification is ambiguous, acknowledge the ambiguity and explain possible interpretations\n"
            "• If the requested information is not found in the provided context, clearly state: "
            "'This information is not covered in the provided specification sections'\n"
            "• When discussing compliance or conformance, reference the specific specification clauses\n"
            "• For technical parameters, include units, tolerances, and conditions as specified\n\n"
            
            "CONTEXT (Specification Document Sections):\n"
            f"{context}\n\n"
            
            f"QUESTION: {query}\n\n"
            
            "SPECIFICATION-BASED RESPONSE:"
        )
    
    def convert_to_markdown(self, 
                           pdf_paths: List[str],
                           output_dir: str = "outputs") -> List[str]:
        """Convert PDFs to markdown format and save to outputs directory."""
        output_path = Path(output_dir)
        converted_files = []
        
        for pdf_path_str in pdf_paths:
            pdf_path = Path(pdf_path_str)
            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                continue
                
            logger.info(f"Converting {pdf_path.name} to markdown...")
            output_file = self.pdf_processor.extract_to_markdown(pdf_path, output_path)
            
            if output_file:
                converted_files.append(output_file)
        
        logger.info(f"Converted {len(converted_files)} PDFs to markdown in {output_dir}/")
        return converted_files
    
    def _save_metadata(self, meta_file: str, metadatas: List[ChunkMetadata]) -> None:
        """Save metadata to NDJSON file."""
        with open(meta_file, 'w', encoding='utf-8') as f:
            for metadata in metadatas:
                f.write(json.dumps(metadata.to_dict(), ensure_ascii=False) + '\n')
    
    def _load_metadata(self, meta_file: str) -> List[ChunkMetadata]:
        """Load metadata from NDJSON file."""
        metadatas = []
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                metadata = ChunkMetadata(
                    pdf_name=data['pdf'],
                    page=data['page'],
                    chunk_id=data['chunk_id'],
                    text=data['text']
                )
                metadatas.append(metadata)
        return metadatas


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Streamlined PDF RAG Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build index from PDFs')
    index_parser.add_argument('--pdf', required=True, nargs='+', help='PDF file paths')
    index_parser.add_argument('--index-file', default='pdf_index.faiss', help='Index output file')
    index_parser.add_argument('--meta-file', default='pdf_meta.ndjson', help='Metadata output file')
    index_parser.add_argument('--model', default='models/all-mpnet-base-v2', help='Embedding model path')
    index_parser.add_argument('--chunk-size', type=int, default=400, help='Chunk size in words')
    index_parser.add_argument('--overlap', type=int, default=50, help='Overlap between chunks')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert PDFs to markdown')
    convert_parser.add_argument('--pdf', required=True, nargs='+', help='PDF file paths')
    convert_parser.add_argument('--output-dir', default='outputs', help='Output directory for markdown files')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the index')
    query_parser.add_argument('--query', required=True, help='Query text')
    query_parser.add_argument('--index-file', default='pdf_index.faiss', help='Index file path')
    query_parser.add_argument('--meta-file', default='pdf_meta.ndjson', help='Metadata file path')
    query_parser.add_argument('--model', default='models/all-mpnet-base-v2', help='Embedding model path')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    query_parser.add_argument('--host', default='127.0.0.1', help='LLM server host')
    query_parser.add_argument('--port', type=int, default=5001, help='LLM server port')
    query_parser.add_argument('--no-llm', action='store_true', help='Skip LLM generation')
    query_parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens for LLM response')
    query_parser.add_argument('--prompt-template', help='Custom prompt template')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'index':
            pipeline = RAGPipeline(model_name=args.model)
            pipeline.build_index(
                pdf_paths=args.pdf,
                index_file=args.index_file,
                meta_file=args.meta_file,
                chunk_size=args.chunk_size,
                overlap=args.overlap
            )
            
        elif args.command == 'convert':
            pipeline = RAGPipeline()  # Don't need model for conversion
            converted_files = pipeline.convert_to_markdown(
                pdf_paths=args.pdf,
                output_dir=args.output_dir
            )
            print(f"\nConverted files:")
            for file_path in converted_files:
                print(f"  ✓ {file_path}")
                
        elif args.command == 'query':
            pipeline = RAGPipeline(model_name=args.model)
            llm_client = None if args.no_llm else LLMClient(args.host, args.port)
            
            result = pipeline.query(
                query=args.query,
                index_file=args.index_file,
                meta_file=args.meta_file,
                top_k=args.top_k,
                llm_client=llm_client,
                max_tokens=args.max_tokens,
                custom_prompt=args.prompt_template
            )
            
            if args.no_llm:
                print(result)
                
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()