""" Demonstration of composing a chatbot function using LangchainMethods and RAGData
classes.

The correct way to split this is to implement free functions for each respective concept:
* Parsing
* vector database
* LLM model
* etc

But as a demo, all the free functions wrapper class, and settings are defined in one place.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import joblib
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from llama_parse import LlamaParse

from qabot import LangchainMethods, RAGData, langchain_chatbot_factory


def parse_with_llamaparse(input: Path, output: Path, **kwargs):
    """

    Note, this could be refactored away - just store the parsed data
    as MD on disk (which needs to be done anyway).

    :return:
    """
    input = Path(input)
    output = Path(output)

    if output.is_file():
        print(f"Reading cached, parsed output from disk: {output.as_posix()}")
        return joblib.load(output)

    # Parse then cache
    parser = LlamaParse(**kwargs)
    llama_parsed_docs = parser.load_data(input.as_posix())
    joblib.dump(llama_parsed_docs, output)

    return llama_parsed_docs


def dump_parsed_document(llama_docs: list, file_minus_ext: Path) -> Path:
    """

    UnstructuredMarkdownLoader reads a file from disk, and not a stream,
    so write the parsed document/s to markdown

    :return:
    """
    file = file_minus_ext.with_suffix('.md')
    with open(file, 'w') as f:
        for doc in llama_docs:
            f.write(doc.text + '\n')
    return file


def chunk_document(markdown_path, **kwargs) -> List[Document]:
    """
    Modify to pass **kwargs for chunking
    :param markdown_path:
    :return:
    """
    loader = UnstructuredMarkdownLoader(markdown_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(**kwargs)
    docs = text_splitter.split_documents(documents)

    # TODO Change this to a logger
    print(f"Number of documents loaded: {len(documents)}")
    print(f"Total number of document chunks generated :{len(docs)}")

    return docs


def vector_store_database(vs_path, chunked_docs, embed_model: Embeddings, **kwargs):
    """
    Could refactor by just passing vs_path and **kwargs
    :param vs_path:
    :param embed_model:
    :param chunked_docs:
    :return:
    """
    vs_path = Path(vs_path)
    kwargs.update({'persist_directory': vs_path.as_posix(),  # Local mode with in-memory storage only
                   'collection_name': "rag"})

    if vs_path.is_dir():
        # Note, not sure if/why this is a different keyword for the constructor verse
        # from_documents method
        kwargs.update({'embedding_function': embed_model})
        print(f"Loading existing Chroma database from {vs_path.as_posix()}")
        vs = Chroma(**kwargs)
    else:
        kwargs.update({'documents': chunked_docs,
                       'embedding': embed_model})
        print(f"Generating Chroma database at {vs_path.as_posix()}")
        vs = Chroma.from_documents(**kwargs)
    return vs


# TODO(Alex)
# Once it works, try using the type constructor to avoid
# writing so much for PrototypeBotMethods
class PrototypeBotMethods(LangchainMethods):

    @staticmethod
    def parse_document(input: Path, output: Path, **kwargs):
        return parse_with_llamaparse(input, output, **kwargs)

    @staticmethod
    def dump_parsed_document(llama_docs, output_without_extension: Path):
        return dump_parsed_document(llama_docs, output_without_extension)

    @staticmethod
    def chunk_document(file, **kwargs):
        return chunk_document(file)

    @staticmethod
    def embedding_model(embed_model_name) -> Embeddings:
        return FastEmbedEmbeddings(model_name=embed_model_name)

    @staticmethod
    def vector_store_database(vs_path, chunked_docs, embedding_model, **kwargs):
        return vector_store_database(vs_path, chunked_docs, embedding_model, **kwargs)

    @staticmethod
    def vector_store_to_retriever(vs, **kwargs):
        return vs.as_retriever(**kwargs)

    @staticmethod
    def llm_model_constructor(**kwargs):
        return Ollama(**kwargs)

    @staticmethod
    def set_prompt_template(prompt_template: str) -> PromptTemplate:
        return PromptTemplate(template=prompt_template,
                              input_variables=['context', 'question']
                              )


if __name__ == "__main__":

    # Chatbot to interaction with the DFTB+ manual

    # Parser options
    parsing_instruction = """The provided document is a manual for using a density-functional tight-binding theory 
    code, DFTB+. This provides descriptions on all input variables, and valid input formats for the code.
    It contains many tables, and the description of the custom structured data input format for DFTB+.
    Try to be precise while answering the questions.
    """

    llama_parse_opts = {'api_key': os.environ.get('LLAMA_CLOUD_API_KEY'),
                        'result_type': "markdown",
                        'parsing_instruction': parsing_instruction,
                        'max_timeout': 5000,
                        'verbose': True
                        }

    # # This could be improved by using ollama-specific prompts
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    settings = {'parser_input': Path('inputs/dftb_manual.pdf'),
                'parser_output': Path('data/dftb_llama.pk'),
                'parser_options': llama_parse_opts,
                'chunk_options': {'chunk_size': 2000, 'chunk_overlap': 100},
                'embed_model_name': "BAAI/bge-base-en-v1.5",
                'vs_path': "chroma_db_llamaparse1",
                'vs_options': {},
                'vs_retriever_options': {'search_kwargs': {'k': 3}},
                'llm_options':  {'temperature': 1, 'model': "llama3"},
                'prompt_template': prompt_template
                }

    chatbot = langchain_chatbot_factory(PrototypeBotMethods(), RAGData(**settings))

    print("Just prior to querying")

    response = chatbot("what version of DFTB+ is this manual for?")
    print(response)

    response = chatbot("Write me Geometry and  LatticeVectors inputs for GaAs")
    print(response)
