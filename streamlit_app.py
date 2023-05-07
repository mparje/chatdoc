import streamlit as st
from langchain.tools import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain
from langchain.chat_models import ChatOpenAI
import os, asyncio, trafilatura
from langchain.docstore.document import Document
import openai

def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )

class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "Browse a webpage and retrieve the information relevant to the question."
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)
    qa_chain: BaseCombineDocumentsChain

    def _run(self, question: str, url: str) -> str:  # Orden de argumentos corregido aquí
