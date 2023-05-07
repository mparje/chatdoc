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

    def _run(self, question: str, url: str) -> str:
        result = trafilatura.extract(trafilatura.fetch_url(url))
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError

# Streamlit app
st.title("Consulta en sitio web")

api_key = st.sidebar.text_input("Introduce tu clave de API de OpenAI", type="password")
if not api_key:
    st.warning("Por favor, introduce una clave de API válida para continuar.")
else:
    openai.api_key = api_key
    url = st.text_input("URL del sitio web:", "https://uuki.live/")
    question = st.text_input("Pregunta:")

    llm = ChatOpenAI(temperature=1.0, openai_api_key=api_key)
    query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def run_tool(url: str, question: str):
        return query_website_tool.run(question, url)

    if url and question:
        answer = run_tool(url, question)
        st.write("Respuesta:", answer)
