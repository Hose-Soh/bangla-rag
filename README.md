
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

---
## Q/A

1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

    Used bangla_pdf_ocr library to extract text from the PDF files. This library is specialized for processing PDFs that contain Bangla text, leveraging OCR techniques tailored for Bangla script. Standard PDF text extraction methods often struggle with non-Latin scripts or scanned documents. The bangla_pdf_ocr package provides a more accurate and reliable way to convert scanned Bangla PDFs into clean text, which is essential for downstream NLP tasks. The raw PDF extraction sometimes included extraneous content like page numbers, English characters, digits, hyphens, and unwanted whitespace. To address this, we implemented a custom cleaning step in data_processing.py that extracts specific page content ranges and removes irrelevant characters using regular expressions. This helped produce cleaner, more relevant text chunks.

2. What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

    Used a character limit-based chunking strategy with overlap, implemented via LangChain's RecursiveCharacterTextSplitter. Specifically, documents are split into chunks of about 1000 characters with 100 characters of overlap between chunks. This approach balances granularity and context retention. Each chunk is large enough to preserve semantic coherence but small enough to keep retrieval efficient. The overlap ensures that important context spanning chunk boundaries is retained, reducing the risk of losing information relevant to user queries. Character-based chunking works well because it avoids dependency on perfect paragraph or sentence detection, which can be unreliable in noisy OCR outputs. The overlap improves semantic continuity, helping the retrieval model find meaningful matches even if the query relates to context near chunk edges.

3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

    Used the "l3cube-pune/bengali-sentence-similarity-sbert" model from HuggingFace embeddings. This model is specifically trained on Bengali sentence similarity tasks, making it well-suited for capturing semantic relationships in Bangla text. Its embeddings are optimized to represent the meaning of Bangla sentences in a dense vector space, allowing for effective comparison of queries and documents.
    The SBERT-based model encodes sentences into fixed-size vectors such that semantically similar sentences are close together in the embedding space. This facilitates meaningful similarity computations that go beyond keyword matching and understand the context and nuances of the text.

4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

    The query and document chunks are compared using cosine similarity within a Pinecone vector index. We use a retrieval approach (RetrievalQA with map_reduce chain) where the most semantically similar chunks are retrieved based on their embeddings' cosine similarity to the query embedding. Cosine Similarity is widely used for vector similarity as it normalizes for vector magnitude and focuses on direction, which works well for dense semantic embeddings. Pinecone Vector Store is chosen for its scalability, speed, and managed infrastructure for vector similarity search, enabling efficient retrieval over large document collections. The setup integrates with LangChain, making it easier to build complex retrieval and QA pipelines.

5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

    We rely on the embedding model trained for semantic similarity in Bangla, so both the question and document chunks are encoded into the same semantic space. Additionally, the map_reduce retrieval approach aggregates relevant chunks and summarizes answers, which helps reduce noise from irrelevant chunks. The system might retrieve less relevant or overly general chunks, leading to less precise answers if the query is vague or missing context. Since the prompt instructs the model to answer normally when out of context, it attempts to provide a best-effort response even without relevant document grounding. Additional chain can be added to answer out of context question. And SQl chain like sqllama2 can be added to capture the table formatted text perfectly.

6. Do the results seem relevant? If not, what might improve them (e.g., better chunking, better embedding model, larger document)?

    The agent was struggling at first with the older openai and gemini models and with the lower "most matched documents: q" number that was set up. It was struggling too because of a lower chunk size at first, then I had to increase it. But, the gemini-2.5-pro and gpt-3.5-turbo (and higher models) can perfectly retrive the correct answer. Unfortunately, free tiers of these models are very limited in case of token size and how much they can be used. As a result, we could not test many questions. Only tested the given question and the expected result. These models are absolutely spot on in case of these questions. One last thing is that OpenAI models are faster to retrive answer than gemini models. The agent still takes lots of time to answer because of we are retrieving a high number of matched documents. With the proper tuning of the parameters of different functions, this response time can be decreased. 

---
