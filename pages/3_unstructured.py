import streamlit as st

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

from openai import OpenAI

from pinecone import Pinecone

from unstructured.staging.base import dict_to_elements

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import hashlib


from utils import show_navigation
show_navigation()

client=OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
PINECONE_INDEX_NAME=st.secrets['PINECONE_INDEX_NAME']

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

def process_file(file_contents, file_name):
    s=UnstructuredClient(api_key_auth=st.secrets['UNSTRUCTURED_API_KEY'])

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

    tables = [el for el in elements if el.category == "Table"]
    st.write("# START")
    final_text=""
    for t in tables:
        table_html = t.metadata.text_as_html
        final_text += table_html
        st.write(table_html)
    st.write("# COMPLETE")
    nontables = [el for el in elements if el.category != "Table"]
    for n in nontables:
        final_text += n.text
    return resp, elements, tables, final_text

def get_model():
    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=st.secrets['OPENAI_API_KEY'])
    return model

a = """
def process_query(table_data):
    st.write("# Answer question based on table data")
    if query := st.text_input("What do you want to know?"):
        model=get_model()
        template = ""Answer the question based only on the following context:
                {context}

                Question: {question}
        ""
        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()
        chain =  prompt | model | output_parser
        resp=chain.invoke({"context": table_data, "question": query})
        return resp
"""

#
# Main
#

st.write("# Welcome to Streamlit! 👋")
st.markdown("# Upload file with table: PDF")
uploaded_file=st.file_uploader("Upload PDF file",type="pdf")
if uploaded_file is not None:
    file_contents = uploaded_file.getbuffer()
    file_name = uploaded_file.name
    resp, elements, tables, final_text = process_file(file_contents, file_name)
    embedding = embed(final_text,uploaded_file.name)
    with st.sidebar.expander("pdfcontent"):
        st.write(final_text)
    #final_resp=process_query(final_text)
    #st.write(final_resp)