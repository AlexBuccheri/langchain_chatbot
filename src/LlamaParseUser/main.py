""" Parse pdf and feed into an LLM
"""
import os
from pathlib import Path

import joblib
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from llama_parse import LlamaParse
import nest_asyncio  # noqa: E402

# from groq import Groq
# from langchain_groq import ChatGroq

# Enabled nested asyncio Event Loops (see above for more notes)
# nest_asyncio.apply()

# API keys
llamaparse_api_key = os.environ.get('LLAMA_CLOUD_API_KEY')
groq_api_key = os.environ.get("GROQ_API_KEY")


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


def chunk_document(llama_docs):
    """
    Creates a vector database using document loaders.
    Load markdown and split into chunks

    :param llama_docs
    :return docs
    """
    # API for langchain is such that one needs to pass it a file path
    # rather than a stream of some sort
    markdown_path = "data/output.md"
    with open(markdown_path, 'w') as f:
        for doc in llama_docs:
            f.write(doc.text + '\n')

    # Implicitly depends on unstructured package
    loader = UnstructuredMarkdownLoader(markdown_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print(f"Number of documents loaded: {len(documents)}")
    print(f"Total number of document chunks generated :{len(docs)}")

    return docs


if __name__ == "__main__":

    if llamaparse_api_key is None:
        raise ValueError("LLAMA_CLOUD_API_KEY not found in environment")

    if groq_api_key is None:
        raise ValueError("GROQ_API_KEY not found in environment")

    # Parse DFTB+ manual with Llama parse
    dftb_output = Path("data/dftb_llama.pk")

    if dftb_output.is_file():
        print(f"Reading already-parsed file from disk: {dftb_output.as_posix()}")
        llama_docs = joblib.load(dftb_output)
    else:
        llama_docs = parse_dftb_with_llama()
        joblib.dump(llama_docs, dftb_output)

    docs = chunk_document(llama_docs)

    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Create a Chroma vector database from the chunked documents
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_llamaparse1",  # Local mode with in-memory storage only
        collection_name="rag"
    )

    # TODO(Alex) Replace this step
    # Creates a new ChatGroq object named chat_model
    # Sets the temperature parameter to 0, indicating that the responses should be more predictable
    # Sets the model_name parameter to “mixtral-8x7b-32768“, specifying the language model to use

    # chat_model = ChatGroq(temperature=0,
    #                       model_name="mixtral-8x7b-32768",
    #                       api_key=userdata.get("GROQ_API_KEY")
    #                       )
