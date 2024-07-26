""" Generic implementation of a langhchain query-response bot.

Method API are defined according to `LangchainMethods`
Data are defined by `RAGData`

An instance of a specialised child class of `LangchainMethods` provides the methods
to `langchain_chatbot_factory`. Options for each operation used in defining the
query-response bot is supplied by an instance of `RAGData`
"""
from __future__ import annotations

import abc
from abc import ABC
from pathlib import Path
from typing import Callable, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGData:
    # Parser
    parser_input: Path
    parser_output: Path
    parser_options: dict
    # Chunking
    chunk_options: dict
    # Embedding
    embed_model_name: str
    # Vector store
    vs_path: str | Path
    vs_options: dict = {}
    vs_retriever_options: dict
    # LLM
    llm_options: dict
    # Chatbot
    prompt_template: str

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Ensure all paths are of type Path
        path_keys = ['parser_input', 'parser_output', 'vs_path']
        for key in path_keys:
            self.__dict__[key] = Path(self.__dict__[key])


class LangchainMethods(ABC):
    """Methods of a langchain Q/A bot

    Function signatures, no state.
    """

    @staticmethod
    @abc.abstractmethod
    def parse_document(input: Path, output: Path, **kwargs) -> list:
        """ Parse file (.pdf, .html, etc) into a structured file type

        :param input: File to parse.
        :param output: Cached, parsed file.
        :param kwargs: Options for the parser.
        :return: List of parsed documents.
        """
        pass

    @staticmethod
    def dump_parsed_document(*args) -> None | Path:
        """ Some loaders require a file on disk, rather than a file stream.

        If args are passed, expect:
            parsed_doc: list
            output_without_extension: Path

        where one must implement in the routine
        file = output_without_extension.with_suffix('.md')
        for example.

        If dumping is not necessary, do not implement in the child class.

        :return: None or Path
        """
        return None

    @staticmethod
    @abc.abstractmethod
    def chunk_document(parsed_doc, **kwargs) -> List[Document]:
        """ Chunk a parsed document.

        :param parsed_doc: parsed document data, or the file path to that data.
        param kwargs: chunk function options.
        :return:
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def embedding_model(embed_model_name: str) -> Embeddings:
        """ Return an embedding model instance.

        Could change this to kwargs

        param: embed_model_name: Embedding model name
        :return:
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def vector_store_database(vs_path: Path, chunked_docs, embedding_model, **kwargs):
        """ Generate or read a vector store database

        :param vs_path: persist_directory
        :param chunked_docs: Chunked documents returned from `chunk_document` method
        :param embedding_model: Embedding model returned from `embedding_model` method
        :param kwargs: Additional optional args for the vector store constructor.
        :return: vs: Vector store database instance.
        """

    @staticmethod
    @abc.abstractmethod
    def vector_store_to_retriever(vs, **kwargs):
        """ Convert a vector store to a retriever

        :param vs: Vector store instance
        :return:
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def llm_model_constructor(**kwargs):
        """ Initialise an LLM

        :param kwargs:
        :return:
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def set_prompt_template(prompt_template: str) -> PromptTemplate:
        """Prompt template for a language model.

        A prompt template consists of a string template. It accepts a set of parameters
        from the user that can be used to generate a prompt for a language model.

        The template can be formatted using either f-strings (default) or jinja2 syntax.
        :return:
        """


def query_chatbot_func_factory(qa: RetrievalQA) -> Callable[[str], str]:
    """ Returns a function to query a chatbot.

    :param qa: Question-answering chain instance
    :return: query_func: Function that takes a string query and
    returns a string response.
    """

    def query_func(query: str):
        response = qa.invoke({"query": query})
        return response['result']

    return query_func


def langchain_chatbot_factory(lcm: LangchainMethods, data: RAGData) -> Callable[[str], str]:
    """ Generic procedure to spin up a chatbot (retrieval QA) using langhchain API.

    :return: chat_bot_func: Function that takes a string query and
    returns a string response using an LLM with RAG.
    """
    parsed_doc = lcm.parse_document(data.parser_input, data.parser_output, **data.parser_options)

    output_minus_extension = data.parser_output.parent / data.parser_output.stem
    file_dumped = lcm.dump_parsed_document(parsed_doc, output_minus_extension)

    if file_dumped:
        chunked_doc = lcm.chunk_document(file_dumped)
    else:
        chunked_doc = lcm.chunk_document(parsed_doc)

    embed_model = lcm.embedding_model(data.embed_model_name)

    vs = lcm.vector_store_database(data.vs_path, chunked_doc, embed_model, **data.vs_options)

    retriever = lcm.vector_store_to_retriever(vs, **data.vs_retriever_options)

    llm_model = lcm.llm_model_constructor(**data.llm_options)

    prompt = lcm.set_prompt_template(data.prompt_template)

    # Instantiate the Retrieval Question Answering Chain
    # NOTE, this is depreciated - should replace
    # See https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html
    qa_chain = RetrievalQA.from_chain_type(llm=llm_model,
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": prompt})

    chat_bot_func = query_chatbot_func_factory(qa_chain)
    return chat_bot_func
