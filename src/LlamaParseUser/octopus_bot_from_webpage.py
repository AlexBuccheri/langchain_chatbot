""" Parse the Octopus documentation from its online website
and create a RAG layer

Notes on embedding vectors with langchain:

* Fast embedding documentation says parallel=0 and batch_size=512 are valid,
  but they're not accepted by the langhcain wrapper class
  - https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.fastembed.FastEmbedEmbeddings.html

* Switched to faster model: https://qdrant.github.io/fastembed/examples/Supported_Models/
  but needed to rebuild the db: "BAAI/bge-small-en-v1.5

"""
from pathlib import Path
import pickle
import time
from typing import Callable, List, Tuple

import chromadb
from langchain import hub
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.documents.base import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.html import partition_html

# Required once, to get dependencies for partition_html
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

# Dirty, hard-coded project root
root = Path("/Users/alexanderbuccheri/Codes/LlamaParseUser")


def depreciated_langchain_chatbot_factory(vs, chat_model) -> Callable[[str], str]:
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

    retriever = vs.as_retriever(search_kwargs={'k': 1})

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    qa = RetrievalQA.from_chain_type(llm=chat_model,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=False,  # Reference where responses obtained from
                                     chain_type_kwargs={"prompt": prompt})

    return lambda query: qa.invoke({"query": query})


def langchain_chatbot_factory(vs, llm) -> Callable[[str], str]:
    """
    https://python.langchain.com/v0.2/docs/versions/migrating_chains/retrieval_qa/

    The prompt can be pulled from  https://smith.langchain.com/hub/rlm/rag-prompt
    however I construct manually.

    :param vs:
    :param llm:
    :return:
    """
    # prompt = hub.pull("rlm/rag-prompt")
    template = PromptTemplate(input_variables=['context', 'question'],
                              template="You are an assistant for question-answering tasks. "
                                       "Use the following pieces of retrieved context to answer the question. "
                                       "If you don't know the answer, just say that you don't know. "
                                       "Keep the answer concise where possible.\n"
                                       "Question: {question} \nContext: {context} \nAnswer:"
                              )

    prompt = ChatPromptTemplate(input_variables=['context', 'question'],
                                messages=[HumanMessagePromptTemplate(prompt=template)]
                                )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
            {"context": vs.as_retriever() | format_docs,
             "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return lambda query: qa_chain.invoke({"query": query})


def get_or_create_chroma_vectorstore(db_path: Path, embed_model, docs=None):
    """ Get or create a chroma vectorstore
    """
    if docs is None:
        if not db_path.is_dir():
            raise FileNotFoundError("Database does not exist")

        print(f"Loading existing Chroma database from {db_path.as_posix()}")

        # This is slow because of the interbal call to construct embedding vectors
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


def add_to_chroma_vectorstore(chunked_documents: List[LCDocument], vs: Chroma, max_batch_size=5000):
    """ Add entries to an existing vector store.

    This batches a call to chroma's upsert.
    If the id is already present, it will get replaced.

    Written my own batched function because langchain_community.vectorstores.chromo.Chroma
    does not wrap the full functionality of chroma DB:

        * from_documents is a light wrapper over from_texts, which creates a Chroma instance
          - I **think** this should only be used for new Chroma instances
        * from_texts also creates a new Chroma instance, and has a batched call to add_texts
        * add_texts calls upsert
        * However, do any of these calls work on adding documents to an existing database?

        * update_documents is batched, but calls update in the generic parent class. Assuming this wraps
          chroma's update, an error will be logged and the update will be ignored if an
          id is not found in the collection.

        * add_documents is defined in the VectorStore base class of Chroma, and this has no batching
         (as it obviously does not know about chroma at this level), **suggesting** there's no add_documents
         method

        So it's not clear to me that a batched call to upsert can be made on an existing collection.
        Langchain's API wrapping is quite opaque.

    Computing the embedding vectors is definitely the bottleneck, and is why it **looks** like
    chroma is slow. It's not, it's that Chroma hides/include the call to evaluating the
    embedding vectors to pass to the store. Hence, another reason for exposing the calls

    How I would distribute computing the embedding vectors with MPI4PY. In the batched loop below
    add:

    ```python
    for i1, i2 in batches:
        batched_texts = texts[i1:i2]
        ev_batches = distribute_loop_mpi(i1, i2, n_processes)
        j1, j2 = ev_batches[rank]
        local_embeddings = vs._embedding_function.embed_documents(batched_texts[j1: j2])
        embeddings = sum(comm.allgather(local_embeddings), [])
   ```
    One could instead just distribute the batches over MPI processes, so each one does
    len(batches) / n_processes but this will mean batches that are small for chroma.

    :param vs: Chroma vectorstore to add or modify documents in
    :param max_batch_size: Max batch size for chroma, which should not exceed ~5400
    else an exception is raised
    :return:
    """
    assert max_batch_size < 5400, 'Chroma does not accept batches >~ 5400'
    print("Extending the Octopus vectorstore")

    # Adapted from the `update_documents` method of Chroma class in langchain
    texts = [doc.page_content for doc in chunked_documents]
    metadatas = [doc.metadata for doc in chunked_documents]

    # Create batches for chroma DB
    # Adapted from from chromadb.utils.batch_utils import create_batches
    # Cleaner to return the indices, than a bunch of optionally-filled tuples
    batches = batch_loop(len(ids), max_batch_size)

    for i1, i2 in batches:
        print(f'Embedding text for batch {i1}:{i2}')

        start_time_em = time.time()
        embeddings = vs._embedding_function.embed_documents(texts[i1:i2])
        end_time_em = time.time()
        print("Embedding time", end_time_em - start_time_em)

        vs._collection.upsert(
            ids=ids[i1:i2],
            embeddings=embeddings,
            metadatas=metadatas[i1:i2],
            documents=texts[i1:i2]
        )

    return


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


def batch_loop(n: int, batch_size: int) -> List[Tuple[int, int]]:
    batches = []
    for i in range(0, n, batch_size):
        batches.append((i, i + batch_size))
    batches[-1] = (batches[-1][0], n)
    return batches


def distribute_loop_mpi(i1: int, i2: int, n_processes: int) -> List[Tuple[int, int]]:
    dx = int((i2 - i1) / n_processes)
    remainder = (i2 - i1) % n_processes

    mpi_batch = [(0, dx + remainder)]
    remainder -= 1

    for j in range(1, n_processes):
        start = mpi_batch[-1][1]
        end = start + dx + remainder
        mpi_batch.append((start, end))
        if remainder > 0:
            remainder -= 1

    return mpi_batch


if __name__ == "__main__":

    # Parse Octopus webpages
    # with open(root / 'oct_webpages.txt', 'rb') as fid:
    #     urls = pickle.load(fid)
    # urls = sorted(list(urls))

    # Load urls
    with open(root / 'inputs/octopus_urls', 'r') as fid:
        urls = fid.read().strip().split('\n')
    print(f'{len(urls)} unique Octopus urls')

    # Took an hour to do 12, 1000. 1000 entries is ~ 1 GB
    # Not a problem if this is true - will replace any existing db entries
    add_docs = True
    url_start, url_end = 2000, 2001

    # Parse
    documents = parse_documents_from_urls(urls[url_start:url_end])

    # Chunk
    print('Chunking')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Number of documents loaded: {len(documents)}")
    print(f"Total number of document chunks generated :{len(chunked_documents)}")

    # Use the website name appended with -chunk, as the idea,
    # where chunk is the ith chunk
    # Not using `unique_id_from_webpage`, as it's too slow
    id_counters = {url: 0 for url in urls[url_start:url_end]}
    ids = []
    for doc in chunked_documents:
        source = doc.metadata['source']
        ids.append(source + '-' + str(id_counters[source]))
        id_counters[source] += 1

    # Create or load the vector store
    # Note, could host the vectorstore on a server
    vs_name = root / "chroma_db_octopus"
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=8)

    start_time = time.time()
    if add_docs:
        vs = get_or_create_chroma_vectorstore(vs_name, embed_model)
        add_to_chroma_vectorstore(chunked_documents, vs, max_batch_size=5000)
    else:
        print("Creating an Octopus vectorstore")
        vs = get_or_create_chroma_vectorstore(vs_name, embed_model, chunked_documents)
    end_time = time.time()
    print("Chroma.from_documents time (s)", end_time - start_time)
