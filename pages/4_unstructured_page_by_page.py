import streamlit as st
from pinecone import Pinecone
from openai import OpenAI

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import hashlib

from unstructured.staging.base import dict_to_elements

from pathlib import Path

from PyPDF2 import PdfWriter, PdfReader

#from langchain_openai import ChatOpenAI
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.runnables import RunnableParallel, RunnablePassthrough


from utils import show_navigation
import time
show_navigation()

PINECONE_INDEX_NAME=st.secrets['PINECONE_INDEX_NAME']
client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

def embed(text,filename):
    pc = Pinecone(api_key = st.secrets["PINECONE_API_KEY"])
    #pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index = pc.Index(PINECONE_INDEX_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap  = 200,length_function = len,is_separator_regex = False)
    docs=text_splitter.create_documents([text])
    for idx,d in enumerate(docs):
        hash=hashlib.md5(d.page_content.encode('utf-8')).hexdigest()
        embedding=client.embeddings.create(model="text-embedding-ada-002", input=d.page_content).data[0].embedding
        metadata={"hash":hash,"text":d.page_content,"index":idx,"model":"text-embedding-ada-003","docname":filename}
        index.upsert([(hash,embedding,metadata)])
    return



def process_file_direct(file_contents, file_name):
    s=UnstructuredClient(api_key_auth=st.secrets['UNSTRUCTURED_API_KEY'],server_url=st.secrets['UNSTRUCTURED_API_SERVER'])

    files=shared.Files(
        content=file_contents,
        file_name=file_name,
    )

    req = shared.PartitionParameters(
        files=files,
        strategy="hi_res",
        hi_res_model_name="yolox",
        skip_infer_table_types=[],
        pdf_infer_table_structure=True,
    )

    try:
        resp = s.general.partition(req)
        elements = dict_to_elements(resp.elements)
    except SDKError as e:
        print(e)

    print(f"Elements = {elements}")

    final_text=""
    for el in elements:
        if el.category == "Table":
            table_html = el.metadata.text_as_html
            final_text += table_html
            #st.write(table_html)
        else:
            final_text += el.text
            #st.write(el.text)
    return resp, elements, final_text


def process_file_page_by_page(uploaded_file,file_contents, file_name):
    
    inputpdf = PdfReader(uploaded_file)

    full_text = ""
    for i in range(len(inputpdf.pages)):
        start_time = time.time()

        st.write(f"Starting Page: {i+1}")
        outputpdf = PdfWriter()
        outputpdf.add_page(inputpdf.pages[i])
        fname = Path(file_name).stem+"-page%s.pdf" % i
        with open(fname, "wb") as outputStream:
            outputpdf.write(outputStream)
        with open(fname, "rb") as f:
            resp, elements, final_text=process_file_direct(f.read(), fname)
            full_text += final_text

        end_time = time.time()
        execution_time = end_time - start_time
        st.write(f"Finished Page: {i+1}, time: {execution_time}")
    return resp, elements, full_text
#
# Main
#

st.write("# Unstructured page-by-page! 👋")
print("\n\n\n# Unstructured page-by-page! 👋\n\n")
st.markdown("# Upload file with table: PDF")
uploaded_file=st.file_uploader("Upload PDF file",type="pdf")
if uploaded_file is not None:
    file_contents = uploaded_file.getbuffer()
    file_name = uploaded_file.name

    #Call 1
    start_time = time.time()
    resp1, elements1, final_text1 = process_file_page_by_page(uploaded_file,file_contents, file_name)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Page-by-page Execution time: {execution_time} seconds")
    st.write(f"Page-by-page Execution time: {execution_time} seconds")


    #Call 2
    start_time = time.time()
    resp2, elements2, final_text2 = process_file_direct(file_contents, file_name)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Direct Execution time: {execution_time} seconds")
    st.write(f"Direct Execution time: {execution_time} seconds")


    #resp1, elements1, final_text1 = process_file_page_by_page(file_contents, file_name)
    #resp2, elements2, final_text2 = process_file_direct(file_contents, file_name)
    if st.button("Write to Pinecone"):
        embedding = embed(final_text1,uploaded_file.name)
    with st.sidebar.expander("pdfcontent"):
        st.write(final_text1)
    #final_resp=process_query(final_text)
    #st.write(final_resp)
