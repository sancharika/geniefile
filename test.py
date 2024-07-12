from langchain_core.documents import Document
from components import docLoader, functions
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.telegram import text_to_docs



class CustomDocumentLoader:
    """A document loader that processes document content line by line."""

    def __init__(self, document_content: str) -> None:
        """Initialize the loader with the document content as a string.

        Args:
            document_content: The string containing the document content.
        """
        self.document_content = document_content

    def lazy_load(self):
        """A lazy loader that processes the document content line by line.

        Yields documents one by one, where each document represents a line.
        """
        line_number = 0
        print(self.document_content)
        # for line in self.document_content.splitlines():  # Split by line breaks
        #     yield Document(
        #         page_content=line,
        #         metadata={"line_number": line_number, "source": "String Input"},
        #     )
        #     line_number += 1


def load_doc():
    uploaded_file = st.file_uploader("Choose a document file", type=["pdf", "txt", "docx"])
    loader = docLoader()
    return loader.load(uploaded_file)
    """Loads document content from a string and processes it line by line.

    Args:
        document_str: The string containing the document content.

    Returns:
        An iterator yielding Document objects for each line.
    """
def doc_loader(document_str):
    loader = CustomDocumentLoader(document_str)
    return text_to_docs(loader.lazy_load())
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents
# Load document
text = docLoader.load_doc()

jd, jd1 = st.columns(2)
doc=read_doc('file/')
print(doc)
with jd:
    st.write(text)
with jd1:
    st.write(doc)
# Example usage (assuming document_str holds your document content)
# if text:
#     for doc in load_doc(text):
#         # Process each document (line) here
#         print(f"Line {doc.metadata['line_number']}: {doc.page_content}")