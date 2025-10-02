# PDF RAG Pipeline

A modular Python pipeline for retrieval-augmented generation (RAG) with PDF documents.
It provides tools for PDF parsing and chunking, FAISS vector storage, and integration with KoboldCpp/llama.cpp for streamed LLM completions.

---

## Features

* Extracts text from PDFs and splits it into overlapping chunks
* Generates embeddings using [SentenceTransformers](https://www.sbert.net/)
* Stores and retrieves chunks with [FAISS](https://github.com/facebookresearch/faiss)
* Supports streaming completions from KoboldCpp, llama.cpp, or OpenAI-compatible APIs
* Converts PDFs to Markdown format with basic structure preserved

---

## Directory Structure

```
project-root/
│
├── main.py               # Entry point for CLI
├── pdf_index.faiss       # Generated FAISS index
├── pdf_meta.ndjson       # Metadata file for chunks
├── outputs/              # Markdown exports from PDFs
│   └── example.md
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag_pipeline.git
cd rag_pipeline
```

### 2. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

All functionality is available via `main.py` using subcommands.

### Build Index

Create a FAISS index and metadata file from one or more PDFs:

```bash
python main.py index --pdf file1.pdf file2.pdf
```

Arguments:

| Argument       | Type | Default                    | Description                     |
| -------------- | ---- | -------------------------- | ------------------------------- |
| `--pdf`        | list | required                   | Paths to one or more PDF files  |
| `--index-file` | str  | `pdf_index.faiss`          | Output FAISS index file         |
| `--meta-file`  | str  | `pdf_meta.ndjson`          | Metadata file (NDJSON)          |
| `--model`      | str  | `models/all-mpnet-base-v2` | SentenceTransformers model path |
| `--chunk-size` | int  | 400                        | Chunk size in words             |
| `--overlap`    | int  | 50                         | Overlap between chunks          |

---

### Convert PDFs to Markdown

Convert one or more PDFs into Markdown files:

```bash
python main.py convert --pdf file1.pdf file2.pdf --output-dir outputs
```

Arguments:

| Argument       | Type | Default   | Description                   |
| -------------- | ---- | --------- | ----------------------------- |
| `--pdf`        | list | required  | Paths to PDFs                 |
| `--output-dir` | str  | `outputs` | Directory for markdown output |

---

### Query Index

Search the FAISS index and optionally stream a response from an LLM server:

```bash
python main.py query --query "What is the main topic?" --host 127.0.0.1 --port 5001
```

Arguments:

| Argument            | Type | Default                    | Description                                                            |
| ------------------- | ---- | -------------------------- | ---------------------------------------------------------------------- |
| `--query`           | str  | required                   | Query string                                                           |
| `--index-file`      | str  | `pdf_index.faiss`          | Path to FAISS index                                                    |
| `--meta-file`       | str  | `pdf_meta.ndjson`          | Metadata file                                                          |
| `--model`           | str  | `models/all-mpnet-base-v2` | Embedding model path                                                   |
| `--top-k`           | int  | 5                          | Number of chunks to retrieve                                           |
| `--host`            | str  | `127.0.0.1`                | LLM server host                                                        |
| `--port`            | int  | 5001                       | LLM server port                                                        |
| `--no-llm`          | flag | False                      | Skip LLM generation, only return retrieved chunks                      |
| `--max-tokens`      | int  | 512                        | Maximum tokens for LLM response                                        |
| `--prompt-template` | str  | None                       | Custom prompt template (use `{context}` and `{question}` placeholders) |

---

## Example Workflow

1. Index PDFs:

   ```bash
   python main.py index --pdf docs/spec1.pdf docs/spec2.pdf
   ```

2. Query the index:

   ```bash
   python main.py query --query "Summarize the safety requirements"
   ```

3. Convert to Markdown:

   ```bash
   python main.py convert --pdf docs/spec1.pdf
   ```

---

## LLM Server Setup

This tool can connect to:

* [KoboldCpp](https://github.com/LostRuins/koboldcpp)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* Any OpenAI-compatible API

Ensure the LLM server is running before executing `query`.

