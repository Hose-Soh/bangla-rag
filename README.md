
# Bangla RAG: Retrieval-Augmented Generation Chatbot

This repository contains code for a Retrieval-Augmented Generation (RAG) chatbot system designed to answer queries based on Bangla text documents. The system uses a combination of OCR-processed PDFs, LangChain, Pinecone vector database, and the Google Gemini Pro model to provide intelligent responses in both Bangla and English.

---

## Project Structure

```
├── data/
│   ├── HSC26-Bangla1st-Paper.pdf      # Original raw PDF file 
│   └── cleaned_data.txt            	# Cleaned text extracted from PDF
├── .env                      			# Environment variables 
├── requirements.txt          			# Python dependencies
├── api.py                    			# Flask API backend for serving model responses
├── llm.py                    			# Large Language Model (LLM) initialization and query handling
├── data_processing.py        			# PDF processing and text cleaning
├── streamlit.py              			# Streamlit UI for chatbot frontend
├── vector_db.py              			# Pinecone vector database initialization and document embedding
├── test.ipynb                			# Initial experimental notebook (not main code)
```

---

## Features

- Converts Bangla PDFs into cleaned, processable text.
- Creates document embeddings using HuggingFace Bengali sentence similarity model.
- Stores embeddings in a Pinecone vector store for fast similarity search.
- Uses Google Gemini Pro LLM for answer generation.
- Provides a Flask API to serve queries.
- Offers a Streamlit-based chat UI for interaction.
- Supports both Bangla and English language questions.

---

---

## Packages

- langchain_community
- bangla-pdf-ocr
- langchain 
- langchain-pinecone
- pinecone
- langchain-google-genai
- streamlit
- streamlit_chat
- tiktoken
- unstructured
- python-dotenv
- flask

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Hose-Soh/bangla-rag.git
cd bangla-rag
```

2. Create and activate a virtual environment :

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:

```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
```

---

## Usage

### 1. Data Processing and Initialize Vector Database

Run vector_db.py to convert the raw PDF to cleaned text and to setup the vector database to embed and upload documents to Pinecone


### 2. Run Flask API

Run api.py to start the Flask backend server to expose the query API:

```bash
python api.py
```

API endpoint:

- `POST /agent`: Accepts JSON with a `"question"` field and returns an answer.

### 3. Run Streamlit Chatbot UI

Launch the Streamlit app for an interactive chatbot experience:

```bash
streamlit run streamlit.py
```

---

---

## NOTE
If there is any issue running the flask app or streamlit UI the original work can be found in test.ipynb

---
## Code Overview

- **api.py**: Flask app that serves POST requests on `/agent`, forwarding user questions to the LLM pipeline and returning answers.
- **llm.py**: Handles loading the Google Gemini Pro LLM, creating prompt templates, querying the vector store, and generating answers.
- **data_processing.py**: Converts PDFs to text, cleans documents by removing unwanted characters and irrelevant pages.
- **vector_db.py**: Splits cleaned documents into chunks, embeds them using HuggingFace embeddings, and uploads vectors to Pinecone.
- **streamlit.py**: Streamlit front-end UI that sends user input to the Flask API and displays chat messages.
- **requirements.txt**: Lists required Python packages.
- **test.ipynb**: Initially experimented things in the notebook 

---

