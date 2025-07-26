import pinecone, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from data_processing import data_process


def chunk_data(docs, chunk_size=1000, chunk_overlap=100):
    """
    Splits documents into chunks using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def initialize_pinecone(api_key, index_name, dimension=768, cloud="aws", region="us-east-1"):
    """
    Initializes Pinecone, creates index if it doesn't exist.
    """
    pc = pinecone.Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=dimension,
            metric="cosine",
            spec=pinecone.ServerlessSpec(cloud=cloud, region=region)
        )
    
    return pc.Index(index_name)


def embed_and_upload(documents, embeddings, index, index_name):
    """
    Uploads documents to Pinecone index only if it's empty.
    """
    if index.describe_index_stats()["total_vector_count"] == 0:
        PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)


def initialize_db():
    # Load and clean data
    clean_docs = data_process()
    
    # Chunk text
    chunked_docs = chunk_data(clean_docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="l3cube-pune/bengali-sentence-similarity-sbert"
    )

    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY") 
    index_name = "10mins"
    index = initialize_pinecone(api_key, index_name)

    embed_and_upload(chunked_docs, embeddings, index, index_name)

    # Instantiate Vector Store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    return vectorstore


