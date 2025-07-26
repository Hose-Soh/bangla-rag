import os
import re
from bangla_pdf_ocr import process_pdf
from langchain_community.document_loaders import DirectoryLoader


def pdf_to_text(pdf_path: str, txt_output_path: str):
    """
    Processes the PDF if the output text file does not already exist.
    """
    if not os.path.exists(txt_output_path):
        process_pdf(pdf_path, txt_output_path)


def load_and_clean_documents(folder_path: str, file_pattern: str = "**/*.txt"):
    """
    Loads all text documents from a directory and cleans them.
    Removes English characters, digits, hyphens, and extra whitespace.
    """
    loader = DirectoryLoader(folder_path, glob=file_pattern)
    documents = loader.load()

    for doc in documents:
        content = doc.page_content
        match = re.search(r'Page 6(.*?)Page 20', content, re.DOTALL)
        content = match.group(1)
        content = re.sub(r'[a-zA-Z0-9\-\\n]', '', content)
        content = re.sub(r'\s+', ' ', content)
        doc.page_content = content.strip()
    
    return documents


def data_process():
    pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
    txt_output_path = "data/cleaned_data.txt"

    pdf_to_text(pdf_path, txt_output_path)

    cleaned_docs = load_and_clean_documents("data")

    return cleaned_docs


