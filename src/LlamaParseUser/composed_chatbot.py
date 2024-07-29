""" Demonstration of composing a chatbot function using LangchainMethods and RAGData
classes.

The correct way to split this is to implement free functions for each respective concept:
* Parsing
* vector database
* LLM model
* etc

But as a demo, all the free functions wrapper class, and settings are defined in one place.

TODO
* Add timing decorators that append info to the logger
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import logging
import os
from pathlib import Path
from typing import List, Callable

import joblib
import yaml
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


def redirect_stdout_to_log(func: Callable) -> Callable:
    def modified_func(*args, **kwargs):
        new_stdout = io.StringIO()
        with redirect_stdout(new_stdout):
            return_data = func(*args, **kwargs)
        captured_output = new_stdout.getvalue()
        logger.info(f'Captured stdout: {captured_output}')
        return return_data
    return modified_func


def parse_with_llamaparse(input: Path, output: Path, **kwargs):
    """

    Note, this could be refactored away - just store the parsed data
    as MD on disk (which needs to be done anyway).

    :return:
    """
    input = Path(input)
    output = Path(output)
    kwargs['api_key'] = os.environ.get('LLAMA_CLOUD_API_KEY')

    if kwargs['api_key'] is None:
        raise EnvironmentError('LLAMA_CLOUD_API_KEY not defined as an ENV VAR')

    if output.is_file():
        logger.info(f"Reading cached, parsed output from disk: {output.as_posix()}")
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

    logger.info(f"Number of documents loaded: {len(documents)}")
    logger.info(f"Total number of document chunks generated :{len(docs)}")

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
        logger.info(f"Loading existing Chroma database from {vs_path.as_posix()}")
        vs = Chroma(**kwargs)
    else:
        kwargs.update({'documents': chunked_docs,
                       'embedding': embed_model})
        logger.info(f"Generating Chroma database at {vs_path.as_posix()}")
        vs = Chroma.from_documents(**kwargs)
    return vs


# class PrototypeBotMethods(LangchainMethods):
#
#     @staticmethod
#     def parse_document(input: Path, output: Path, **kwargs):
#         return parse_with_llamaparse(input, output, **kwargs)
#
#     @staticmethod
#     def dump_parsed_document(llama_docs, output_without_extension: Path):
#         return dump_parsed_document(llama_docs, output_without_extension)
#
#     @staticmethod
#     def chunk_document(file, **kwargs):
#         return chunk_document(file)
#
#     @staticmethod
#     def embedding_model(embed_model_name) -> Embeddings:
#         return FastEmbedEmbeddings(model_name=embed_model_name)
#
#     @staticmethod
#     def vector_store_database(vs_path, chunked_docs, embedding_model, **kwargs):
#         return vector_store_database(vs_path, chunked_docs, embedding_model, **kwargs)
#
#     @staticmethod
#     def vector_store_to_retriever(vs, **kwargs):
#         return vs.as_retriever(**kwargs)
#
#     @staticmethod
#     def llm_model_constructor(**kwargs):
#         return Ollama(**kwargs)
#
#     @staticmethod
#     def set_prompt_template(prompt_template: str) -> PromptTemplate:
#         return PromptTemplate(template=prompt_template,
#                               input_variables=['context', 'question']
#                               )


# Globally scoped logger
logger = logging.getLogger('chatbot')


def initialise_logger(level=logging.INFO):
    """ Initialise global logger

    :param level:
    :return:
    """
    levels = [logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING,
              logging.WARN, logging.INFO, logging.DEBUG, logging.NOTSET]

    if level not in levels:
        raise ValueError(f'Logging level invalid: {level}')

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


if __name__ == "__main__":

    # Chatbot to interaction with the DFTB+ manual
    initialise_logger()

    # Parse options and configuration
    project_root = Path(__file__).parents[2]
    with open(project_root / 'tests/dftbplus.yml', 'r') as file:
        config = yaml.safe_load(file)
    settings = config['settings']

    # Use the type constructor to create the class
    PrototypeBotMethods = type(
        'PrototypeBotMethods',
        (LangchainMethods,),
        {
            'parse_document': staticmethod(parse_with_llamaparse),
            'dump_parsed_document': staticmethod(dump_parsed_document),
            'chunk_document': staticmethod(chunk_document),
            'embedding_model': staticmethod(redirect_stdout_to_log(
                lambda name: FastEmbedEmbeddings(model_name=name))),
            'vector_store_database': staticmethod(vector_store_database),
            'vector_store_to_retriever': staticmethod(lambda vs, **kwargs: vs.as_retriever(**kwargs)),
            'llm_model_constructor': staticmethod(lambda **kwargs: Ollama(**kwargs)),
            'set_prompt_template': staticmethod(lambda template:
                                                PromptTemplate(template=template,
                                                               input_variables=['context', 'question'])
                                                ),
        }
    )

    chatbot = langchain_chatbot_factory(PrototypeBotMethods(), RAGData(**settings))

    print("Starting chatbot")

    questions = ["what version of DFTB+ is this manual for?",
                 "Write me Geometry and  LatticeVectors inputs for GaAs"]

    for question in questions:
        print('Question: \n', question, '\n')
        response = chatbot(question)
        print('Response: \n', response, '\n')
