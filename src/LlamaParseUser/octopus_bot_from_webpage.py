""" Parse the Octopus documentation from its online website
and create a RAG layer
"""
import pickle
import time
from pathlib import Path
from typing import Callable, List

import chromadb
import pyshorteners
import requests
from chromadb.utils.batch_utils import create_batches_for_chroma

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents.base import Document as LCDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tqdm import tqdm
from unstructured.partition.html import partition_html

# import nltk
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()


root = Path("/Users/alexanderbuccheri/Codes/LlamaParseUser")


def langchain_chatbot_factory(vs, chat_model) -> Callable[[str], str]:
    """

    :param vs:
    :param chat_model:
    :return:
    """
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    retriever = vs.as_retriever(search_kwargs={'k': 3})

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    qa = RetrievalQA.from_chain_type(llm=chat_model,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": prompt})

    return lambda query: qa.invoke({"query": query})


def get_or_create_chroma_vectorstore(db_path: Path, embed_model, docs=None):
    """ Get or create a chroma vectorstore
    """
    if docs is None:
        if not db_path.is_dir():
            raise FileNotFoundError("Database does not exist")

        print(f"Loading existing Chroma database from {db_path.as_posix()}")

        # This is horrifically slow
        # vs = Chroma(
        #     persist_directory=db_path.as_posix(),
        #     collection_name="octopus_rag",
        #     embedding_function=embed_model
        # )

        # See if explicitly passing the client is faster
        client = chromadb.PersistentClient(path=db_path.as_posix())
        vs = Chroma(
            persist_directory=db_path.as_posix(),
            collection_name="octopus_rag",
            embedding_function=embed_model,
            client=client
        )
    else:
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=db_path.as_posix(),  # Local mode with in-memory storage only
            collection_name="octopus_rag"
        )

    return vs


# No point doing this, as this breaks the rest of the workflow - would need to repackage the database
# using langhchain's Chroma
# client_settings = chromadb.config.Settings(is_persistent=True)
# client = chromadb.Client(client_settings)
# # No idea how to supply embedding to it, from langchain
# collection = client.get_or_create_collection(name="octopus_rag")
# collection.add(texts = [doc.page_content for doc in documents],
#                metadatas = [doc.metadata for doc in documents],
#                ids=[f"id{i}" for i in range(len(metadatas)])


def parse_documents_from_urls(urls: list) -> List[LCDocument]:
    print('Parsing')
    documents = []
    for url in urls:
        print(url)
        try:
            elements = partition_html(url=url)
            # print("\n\n".join([str(el) for el in elements]))
            # Convert to langchain obj
            documents += [LCDocument(page_content=str(element), metadata={"source": url}) for element in elements]
        except ValueError:
            print(f'Skipping {url}')

    return documents


# This is really slow - need a better option... maybe just the url + chunk
url_shortner = pyshorteners.Shortener()


def unique_id_from_webpage(url: str) -> str:
    # Drop the https://tinyurl.com/
    short_url = url_shortner.tinyurl.short(url)[20:]
    return short_url


# This is super-annoying, but assume it won't be used often, or I can rebuild with other ids
def retrive_url(id: str):
    short_url, chunk = id.split('-')
    try:
        response = requests.head("https://tinyurl.com/" + short_url, allow_redirects=True)
        return response.url
    except requests.RequestException as e:
        raise requests.RequestException (f"An error occurred: {e}")


if __name__ == "__main__":

    todos = """    
    1. Make the vector store extendable, then can sequentially add to it
    2. Change chatbot backend to GPT 3.5, or better
        3. Test this by parsing a subset of Octopus webpages and see how it performs
    - Alternatively, look at a more efficient way of parsing the raw data
    - Should be possible as I have all of it
    """
    # Parse Octopus webpages
    # with open(root / 'oct_webpages.txt', 'rb') as fid:
    #     urls = pickle.load(fid)
    # urls = sorted(list(urls))

    with open(root / 'inputs/octopus_urls', 'r') as fid:
        urls = fid.read().strip().split('\n')
    print(f'{len(urls)} unique Octopus urls')

    add_docs = True
    url_start, url_end = 5, 10
    max_batch_size = 5000

    # Idea is to hash the website name, then append with -chunk
    # where chunk is the ith chunk
    id_counters = {url:0 for url in urls[url_start:url_end]}

    # Parse
    documents = parse_documents_from_urls(urls[url_start:url_end])

    # Chunk
    print('Chunking')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Number of documents loaded: {len(documents)}")
    print(f"Total number of document chunks generated :{len(chunked_documents)}")

    # Not using unique_id_from_webpage, as it's too slow
    ids = []
    for doc in chunked_documents:
        source = doc.metadata['source']
        ids.append(source + '-' + str(id_counters[source]))
        id_counters[source] += 1

    # Create or load the vector store
    print('Creating vector store')
    # Docs say  parallel=0, batch_size=512 are valid, but they're not in the code
    # https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.fastembed.FastEmbedEmbeddings.html
    # TODO(Alex) Switch to fast embedding
    # Switched to faster model: https://qdrant.github.io/fastembed/examples/Supported_Models/
    # but would need to rebuild the db: "BAAI/bge-small-en-v1.5
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5", threads=8)
    vs_name = root / "chroma_db_octopus"

    # Note, could host the vectorstore elsewhere
    if add_docs:
        # Should check how new indices are generated when extending a store
        print("Extending the Octopus vectorstore")
        vs = get_or_create_chroma_vectorstore(vs_name, embed_model)

        # Adapted from the `update_documents` method of Chroma class in langchain
        print('Repackaging new langchain Documents')
        texts = [doc.page_content for doc in chunked_documents]
        metadatas = [doc.metadata for doc in chunked_documents]

        # Create batches
        batches = []
        for i in range(0, len(ids), max_batch_size):
            batches.append((i, i + max_batch_size))
        batches[-1] = (batches[-1][0], len(ids))

        # TODO(Alex) Should distribute the batches
        start_time = time.time()
        for i1, i2 in batches:

            print(f'Embedding text for batch {i1}:{i2}')
            start_time_em = time.time()
            embeddings = vs._embedding_function.embed_documents(texts[i1:i2], )
            end_time_em = time.time()
            print("Embedding time", end_time_em - start_time_em)

            vs._collection.upsert(
                    ids=ids[i1:i2],
                    embeddings=embeddings,
                    metadatas=metadatas[i1:i2],
                    documents=texts[i1:i2]
                )

        # for batch in tqdm(create_batches_for_chroma(
        #         api=vs._collection._client,
        #         ids=ids,
        #         metadatas=metadatas,
        #         documents=texts,
        #         embeddings=embeddings
        # )):
        #     vs._collection.upsert(
        #         ids=batch[0],
        #         embeddings=batch[1],
        #         metadatas=batch[2],
        #         documents=batch[3]
        #     )


        # add_documents ultimately wraps `upsert`, with no batching - as defined in a base class
        # This makes it at best slow, or just fail if the number of docs exceeds 5461
        # This above should be added as a new method in the Chroma class
        # vs.add_documents(chunked_documents)
        end_time = time.time()
    else:
        print("Creating an Octopus vectorstore")
        vs = get_or_create_chroma_vectorstore(vs_name, embed_model, chunked_documents)
    print("Chroma.from_documents time (s)", end_time - start_time)

    # TODO Swap this for connection to GPT3.5
    # chat_model = Ollama(temperature=0, model="llama3")
    #
    # chatbot = langchain_chatbot_factory(vs, chat_model)
    #
    # # Should run this in Jupyter
    # print("Starting chatbot")
    #
    # questions = ["what is the latest version of Octopus?",
    #              "What does Octopus do?"]
    #
    # for question in questions:
    #     print('Question: \n', question, '\n')
    #     response = chatbot(question)
    #     # Can also query 'source_documents' to get where this info came from
    #     print('Response: \n', response['result'], '\n')
