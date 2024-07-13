import streamlit as st
import pdfplumber
import docx
import pylatexenc
import base64
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

    def display_doc(self):
        st.markdown("### Document Preview")
        if isinstance(self.uploaded_file, str):
            file_extension = self.uploaded_file.split(".")[-1]
        else:
            file_extension = self.uploaded_file.name.split(".")[-1]

        if file_extension == "pdf":
            if isinstance(self.uploaded_file, str):
                with open(self.uploaded_file, "rb") as file:
                    base64_doc = base64.b64encode(file.read()).decode("utf-8")
            else:
                base64_doc = base64.b64encode(self.uploaded_file.getvalue()).decode("utf-8")

            doc_display = f"""<iframe src="data:application/pdf;base64,{base64_doc}" width="400" height="100%" type="application/pdf"
                                style="height:100vh; width:100%">
                            </iframe>"""

        elif file_extension == "txt":
            text = self.load_txt()
            base64_doc = base64.b64encode(text.encode("utf-8")).decode("utf-8")
            doc_display = f"""<iframe src="data:text/plain;base64,{base64_doc}" width="400" height="100%" style="height:100vh; width:100%">
                            </iframe>"""

        elif file_extension == "docx":
            text = self.load_docx()
            base64_doc = base64.b64encode(text.encode("utf-8")).decode("utf-8")
            doc_display = f"""<iframe src="data:text/plain;base64,{base64_doc}" width="400" height="100%" style="height:100vh; width:100%">
                            </iframe>"""

        elif file_extension == "tex":
            text = self.load_tex()
            base64_doc = base64.b64encode(text.encode("utf-8")).decode("utf-8")
            doc_display = f"""<iframe src="data:text/plain;base64,{base64_doc}" width="400" height="100%" style="height:100vh; width:100%">
                            </iframe>"""

        else:
            doc_display = "Unsupported file format"

        st.markdown(doc_display, unsafe_allow_html=True)
if __name__ == "__main__":
    uploaded_file = st.file_uploader("Choose a document file", type=["pdf", "txt", "docx"])
    docLoader(uploaded_file).load()
