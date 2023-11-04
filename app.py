from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
import urllib
import base64

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

def display_pdf(uploaded_file):
    # Convert the UploadedFile content to base64
    base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}#navpanes=0" width="600" height="900" type="application/pdf" style="border: 1px solid black;"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    #loading env
    if not load_dotenv():
        print("Could not load .env file or it is empty. Please check if it exists and is readable.")
        exit(1)

    #set pages
    st.set_page_config(layout="wide")

    #title
    st.title("AskPDF")
    st.write("---")

    try:
        #upload file
        uploaded_file = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=False)
        #st.write("uploaded_file : ", uploaded_file)
        st.write("---")

        if uploaded_file is not None:
            with st.spinner('Wait for it...'):
                #display
                col1, col2 = st.columns(2)

                with col1:
                    pages = pdf_to_document(uploaded_file)
                        
                    #Split
                    text_splitter = RecursiveCharacterTextSplitter(
                        # Set a really small chunk size, just to show.
                        chunk_size = 1000,
                        chunk_overlap  = 200,
                        length_function = len,
                        is_separator_regex = False,
                    )
                    texts = text_splitter.split_documents(pages)

                    #Embedding
                    embeddings_model = OpenAIEmbeddings()

                    # load it into Chroma
                    db = Chroma.from_documents(texts, embeddings_model)

                    #Question
                    st.header("Ask Questions!")
                    question = st.text_input('Ask Questions with PDF file')

                    if st.button('submit'):
                        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                        qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
                        result = qa_chain({"query": question})
                        st.write(result["result"])

                with col2:
                    display_pdf(uploaded_file)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.button("Reload the page")

if __name__ == "__main__":
    main()
