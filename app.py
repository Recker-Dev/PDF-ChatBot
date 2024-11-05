import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO


load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Access the Google API key from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]

# Configure genai with the API key
genai.configure(api_key=api_key)


 ## Get all the pdfs that are uploaded.
# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader=PdfReader(pdf) ## Pdf is read page-wise, thus a list of pages.
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             text += page.extract_text() or ""  # Safeguard against NoneType

#     return text

def get_pdf_text(pdf_files):
    all_text = ""
    for pdf in pdf_files:
        pdf_bytes = pdf.read()
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            all_text += page.extract_text()
    return all_text

 ## Get chunks of text.
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

## Convert the chunks to vectors
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    ## Converts all the text_chunks to embeddings.
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

## Ask the model
def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer.
    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

## Convert user input to vectors.
def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db=FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) ## Load the DB.
    docs=new_db.similarity_search(user_question) ## Get the part of db relevant to question as new_db.

    chain= get_conversational_chain()


    response=chain(
        {'input_documents':docs , 'question': user_question}
        , return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])



def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with PDF using Gemini 🤷")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF Files and Click on the Submit & Process ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text= get_pdf_text(pdf_docs)
                text_chunks= get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
    