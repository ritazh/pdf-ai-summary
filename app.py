from __future__ import annotations

import streamlit as st
import os

from langchain.agents import tool
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.output_parser import OutputParserException


from azure.storage.blob import BlobServiceClient
from langchain.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from utils.clear_results import with_clear_container

# Azure Blob Storage setup
connection_string = os.environ["STORAGEACCOUNT_CONNECTIONSTRING"]
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "pdfs"

container_client = blob_service_client.get_container_client(container_name)
blobs_list = container_client.list_blobs()
blobs = [blob.name for blob in blobs_list]
#print(blobs)

st.set_page_config(
    page_title="Abori Scientific Publications",
    page_icon="🧑‍🔬",
    initial_sidebar_state="collapsed",
)

"# 🧑‍🔬 Abori Scientific Publications"
"AI to increase Science Outreach"
"Get summary, headlines, and audience!"

# Setup credentials in Streamlit
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to run your own custom questions."
)

# # Add a radio button selector for the model
# model_name = st.sidebar.radio(
#     "Select Model",
#     ("gpt-3.5-turbo", "gpt-4"),
#     help="Select the model to use for the chat.",
# )

@tool
def answer_question(prompt_template: str):
    """Answer a question"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )
    return stuff_chain.run(texts) + "\n\n"


tools = []

if user_openai_api_key:
    openai_api_key = user_openai_api_key
    enable_custom = True
    llm = AzureChatOpenAI(
        openai_api_key=openai_api_key,
        openai_api_version="2023-07-01-preview",
        deployment_name="gpt-4-32k-0613",
        openai_api_type="azure",
        temperature=0,
        streaming=True,
    )

else:
    openai_api_key = "not_supplied"
    enable_custom = False
    llm = AzureChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_version="2023-08-01-preview",
        deployment_name="gpt-4-32k-0613",
        openai_api_type="azure",
        temperature=0,
        streaming=True,
    )

with st.form(key="form"):
    if not enable_custom:
        "Choose one of the papers in the dropdown to get summary, headlines, and target audiences."
    prefilled = (
        st.selectbox(
            "Sample papers",
            sorted([key.replace("_", " ") for key in blobs]),
        )
        or ""
    )
    user_input = ""

    if enable_custom:
        user_input = st.text_input("Or, ask your own question")
    if not user_input:
        user_input = prefilled
    submit_clicked = st.form_submit_button("Submit paper")

output_container = st.empty()

if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="📜")
    
    path_user_input = "_".join(user_input.split(" "))

    if path_user_input in blobs:
        print(f"Getting summary for : {user_input}")
        try:
            storageLoader = AzureBlobStorageContainerLoader(
                conn_str=os.environ["STORAGEACCOUNT_CONNECTIONSTRING"],
                container="pdfs",
                prefix=user_input,
            )
            docs = storageLoader.load()
            texts = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=0)
            texts.extend(text_splitter.split_documents(docs))

            # prompts:

            prompt_template = """Create 3 bulletpoints as headlines for the following:
            "{text}"
            HEADLINES:"""
            answer = answer_question(prompt_template, callbacks=[]) #st_callback, capturing_callback
            answer_container.write("HEADLINES: \n" + answer)

            prompt_template = """Identify three target audiences in Brazil for the following:
            "{text}"
            TARGET AUDIENCE:"""
            answer = answer_question(prompt_template, callbacks=[]) #st_callback, capturing_callback
            answer_container.write("AUDIENCE: \n" + answer)

            prompt_template = """You must extract the following information from the scientific journal article here {text}.  

            Write a 5-7 paragraph news article for a high school student.  

            The first paragraph should describe the results of the scientific journal article.

            The middle paragraphs should go into detail about the scientific journal article.

            Be sure to mention the name of the institution where the research was conducted.

            The last paragraph should describe future work and explain why the research is important.
            CONCISE SUMMARY:"""
            answer = answer_question(prompt_template, callbacks=[]) #st_callback, capturing_callback
            answer_container.write("SUMMARY: \n" + answer)

            
        except OutputParserException as e:
            answer = e.args[0]
