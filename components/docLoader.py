import streamlit as st
import pdfplumber
import docx
import pylatexenc
from langchain_community.document_loaders.telegram import text_to_docs


class docLoader():
    def  __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def load(self):
        if self.uploaded_file is not None:
            st.toast("File uploaded successfully!", icon="✅")

            file_extension = self.uploaded_file.name.split(".")[-1]

            load_functions = {
                "pdf": self.load_pdf,
                "txt": self.load_txt,
                "docx": self.load_docx,
                "tex": self.load_tex
            }

            if file_extension in load_functions:
                text = load_functions[file_extension]()
            else:
                st.write("Unsupported file format")

        else:
            text = ''
        return text

    def load_pdf(self):
        with pdfplumber.open(self.uploaded_file) as pdf:
            pages = pdf.pages
            text = ""
            for page in pages:
                text += page.extract_text()
        return text

    def load_txt(self):
        return self.uploaded_file.getvalue().decode("utf-8")

    def load_docx(self):
        docx_text = docx.Document(self.uploaded_file)
        full_text = [para.text for para in docx_text.paragraphs]
        return "\n".join(full_text)

    def load_tex(self):
        with open(self.uploaded_file.name, 'r') as tex_file:
            tex_content = tex_file.read()
        return pylatexenc.latex2text(tex_content)

def load_doc():
    uploaded_file = st.file_uploader("Choose a document file", type=["pdf", "txt", "docx"])
    loader = docLoader(uploaded_file)
    return loader.load()

if __name__ == "__main__":
    load_doc()