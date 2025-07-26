import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_db import initialize_db
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings


def load_vectorstore():
    """
    Loads the vector store initialized from Pinecone.
    """
    return initialize_db()


def initialize_llm():
    """
    Initializes the Google Gemini Pro model via LangChain wrapper.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.4,
        google_api_key=os.getenv("GOOGLE_API_KEY")  
    )


def create_prompt_templates():
    """
    Returns short-answer map and reduce prompt templates.
    """
    question_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant with the ability to understand Bangla and English. Given a Bangla text as context, you must answer any question 
related to it in both Bangla and English. If the question is unrelated to the Bangla text, answer normally in the appropriate language 
based on the question. Always use the context when relevant, and avoid using it if the question is general or out of context."

Context: {context}
Question: {question}
Answer:
"""
    )

    combine_prompt = PromptTemplate(
        input_variables=["summaries", "question"],
        template="""
Based on the following partial answers, generate a single short one-line answer.

Answers:
{summaries}

Question: {question}
Answer:
"""
    )
    
    return question_prompt, combine_prompt


def create_qa_chain(llm, vectorstore):
    """
    Creates a RetrievalQA chain using a custom prompt and vector store retriever.
    """
    question_prompt, combine_prompt = create_prompt_templates()
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10}),
        chain_type="map_reduce",
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt
        }
    )



def retrieve_answer(chain,  query):
    """
    Performs the full pipeline: retrieve relevant docs and generate an answer.
    """
    # matching_docs = retrieve_query(vectorstore, query)

    response = chain.invoke({"query": query})
    return response


def agent(query):

    #vectorstore = load_vectorstore()
    embeddings = HuggingFaceEmbeddings(model_name="l3cube-pune/bengali-sentence-similarity-sbert")
    vectorstore = PineconeVectorStore.from_existing_index(os.getenv("PINECONE_INDEX_NAME"), embeddings)

    # try:
    #     vectorstore = PineconeVectorStore.from_existing_index(os.getenv("PINECONE_INDEX_NAME"), embeddings)
    # except Exception:
    #     vectorstore = load_vectorstore()

    llm = initialize_llm()
    qa_chain = create_qa_chain(llm, vectorstore)

    response = retrieve_answer(qa_chain, query)
    
    return response



