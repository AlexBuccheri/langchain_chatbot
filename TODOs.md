
## TODOs

* Pin the versions
* Ulimately want to add a frontend (web service/page) and deployment step

## Certificate Issues (venv related?)
For some reason my certificates are out-of-date in this venv, so install `certifi` then export:

```shell
export SSL_CERT_FILE=$(python -m certifi)
export REQUESTS_CA_BUNDLE=$(python -m certifi)
```

SSL_CERT_FILE: Specifies the file to use for verifying SSL connections. 

REQUESTS_CA_BUNDLE: Used by the requests library (which pip uses under the hood) to specify the certificate bundle for 
verifying SSL connections.


## Libraries

* [LLama Parse](https://github.com/run-llama/llama_parse)
  * Parse complex pdfs (and now other docs) into markdown 
  * Primarily support PDFs with tables

* langchain 
  * Wrappers and API for many well-used libraries (including several below) 

* [fastembed](https://github.com/qdrant/fastembed)
  * FastEmbed is a lightweight, fast, Python library built for embedding generation. 
  * We support popular text models. 

* [Chroma](https://github.com/chroma-core/chroma) 
  * The open-source embedding database. 
  * The fastest way to build Python or JavaScript LLM apps with memory!

* [Groq](https://console.groq.com/keys:
  * Run in the cloud on LPUs

* [Ollama](https://github.com/ollama/ollama)
  * API to a range of open-source LLMs. Ollama supports a list of models available on https://ollama.com/library
  * Run locally
  * [Ollama Python Library](https://github.com/ollama/ollama-python)


### Nested asyncio Event Loop Uses

Jupyter Notebooks: 
  * Notebook environments often have their own event loop running, making it impossible to use asyncio.run() directly within 
  a cell.
GUI Applications: 
  * Many GUI frameworks (like Tkinter or PyQt) have their own main event loops, preventing the use of asyncio in a 
  straightforward way.
Web Servers: 
  * Some web server frameworks might already have an asyncio loop running.

