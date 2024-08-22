""" Parse pdf and feed into an LLM
"""
import os
import time
from pathlib import Path
from typing import List

import gradio as gr
import joblib
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from llama_index.core.schema import Document as lammaDocument
from llama_parse import LlamaParse

import nest_asyncio  # noqa: E402


# from groq import Groq
# from langchain_groq import ChatGroq

# Enabled nested asyncio Event Loops (see above for more notes)
# nest_asyncio.apply()

# API keys
llamaparse_api_key = os.environ.get('LLAMA_CLOUD_API_KEY')
groq_api_key = os.environ.get("GROQ_API_KEY")
root = Path("/Users/alexanderbuccheri/Codes/LlamaParseUser")


def parse_dftb_with_llama():
    """ Parse DFTB+ user manual with Llama

    :return:
    """
    dftb_parsing_instruction = """The provided document is a manual for using a density-functional tight-binding theory 
    code, DFTB+. This provides descriptions on all input variables, and valid input formats for the code.
    It contains many tables, and the description of the custom structured data input format for DFTB+.
    Try to be precise while answering the questions.
    """
    parser = LlamaParse(api_key=llamaparse_api_key,
                        result_type="markdown",
                        parsing_instruction=dftb_parsing_instruction,
                        max_timeout=5000,
                        verbose=True
                        )
    llama_parse_documents = parser.load_data("./inputs/dftb_manual.pdf")
    return llama_parse_documents


def chunk_document(llama_docs: List[lammaDocument]):
    """
    Load markdown and split into chunks

    :param llama_docs
    :return docs
    """
    documents = [doc.to_langchain_format() for doc in llama_docs]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print(f"Number of documents loaded: {len(documents)}")
    print(f"Total number of document chunks generated :{len(docs)}")

    return docs


custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def llama_dftb_chat_bot():
    """

    :return:
    """
    if llamaparse_api_key is None:
        raise ValueError("LLAMA_CLOUD_API_KEY not found in environment")

    if groq_api_key is None:
        raise ValueError("GROQ_API_KEY not found in environment")

    # Parse DFTB+ manual with Llama parse
    # Could also try pypdf
    dftb_output = root / "data/dftb_llama.pk"

    if dftb_output.is_file():
        print(f"Reading already-parsed file from disk: {dftb_output.as_posix()}")
        llama_docs = joblib.load(dftb_output)
    else:
        llama_docs = parse_dftb_with_llama()
        joblib.dump(llama_docs, dftb_output)

    docs = chunk_document(llama_docs)

    start_time = time.time()
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    end_time = time.time()
    print("FastEmbedEmbeddings time (s)", end_time - start_time)

    start_time = time.time()

    vector_store = root / "chroma_db_llamaparse1"
    if vector_store.is_dir():
        print(f"Loading existing Chroma database from {vector_store.as_posix()}")
        vs = Chroma(
            persist_directory=vector_store.as_posix(),
            collection_name="rag",
            embedding_function=embed_model
        )
    else:
        # Create a Chroma vector database from the chunked documents
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=vector_store.as_posix(),  # Local mode with in-memory storage only
            collection_name="rag"
        )
    end_time = time.time()
    print("Chroma.from_documents time (s)", end_time - start_time)

    # Convert the vector store into a retriever object, which can be used to search for and retrieve vectors
    # The k parameter determines how many documents to retrieve. Experiment with different values of k to see how
    # it affects performance and accuracy. Try 1 - 3
    start_time = time.time()
    retriever = vs.as_retriever(search_kwargs={'k': 3})
    end_time = time.time()
    print("vs.as_retriever time (s)", end_time - start_time)

    # Assuming Ollama is installed and have llama3 model pulled with `ollama pull llama3
    start_time = time.time()
    chat_model = Ollama(temperature=0, model="llama3")
    end_time = time.time()
    print("Initialising llama3 time (s)", end_time - start_time)

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    # Instantiate the Retrieval Question Answering Chain
    # NOTE, this is depreciated. Should replace as soon as the model works
    # See https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html
    start_time = time.time()
    qa = RetrievalQA.from_chain_type(llm=chat_model,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": prompt})
    end_time = time.time()
    print("RetrievalQA.from_chain_type time (s)", end_time - start_time)

    return qa


def query_chatbot_func_factory(qa):
    """ Returns a function to query a chatbot
    :return:
    """
    def query_func(query: str):
        response = qa.invoke({"query": query})
        return response['result']
    return query_func


if __name__ == "__main__":

    qa = llama_dftb_chat_bot()

    query_chatbot = query_chatbot_func_factory(qa)

    print("Just prior to querying")

    response = query_chatbot("what version of DFTB+ is this manual for?")
    print(response)

    response = query_chatbot("Write me Geometry and  LatticeVectors inputs for GaAs")
    print(response)

    # Basic interface for gradio
    # To run in , run as `gradio src/LlamaParseUser/main.py`
    # Ah, while the machine learning model and all computation continues to run locally on your computer
    # => Could set this all up on my VM
    demo = gr.Interface(
        fn=query_chatbot,
        inputs=["text"],
        outputs=["text"],
    )

    demo.launch()
