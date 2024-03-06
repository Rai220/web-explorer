import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.utilities import GoogleSearchAPIWrapper

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")


def settings():

    # Vectorstore
    from langchain_community.embeddings.gigachat import GigaChatEmbeddings
    from langchain.docstore import InMemoryDocstore
    from langchain_community.vectorstores import Chroma

    embeddings_model = GigaChatEmbeddings(
        base_url="https://beta.saluteai.sberdevices.ru/v1",
        model="Embeddings",
        verify_ssl_certs=False,
        credentials="...",
    )

    vectorstore_public = Chroma(
        collection_name="web",
        embedding_function=embeddings_model,
    )

    # LLM
    # from langchain.chat_models import ChatOpenAI
    from langchain_community.chat_models import GigaChat

    llm = GigaChat(
        streaming=True,
        temperature=0,
        ... only model with funcs needed!!! ...
        verify_ssl_certs=False,
        profanity=False,
        profanity_check=False
    )

    search = GoogleSearchAPIWrapper()

    # Initialize
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm,
        search=search,
        num_search_results=3,
        verify_ssl=False
    )

    return web_retriever, llm


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


st.sidebar.image("img/ai.png")
st.header("`Interweb Explorer`")
st.info(
    "`Я - искусственный интеллект, который может отвечать на вопросы, исследуя, читая и кратко излагая содержание веб-страниц."
    "Меня можно настроить на использование различных режимов: публичного API или приватного (без обмена данными).`"
)

# Make retriever and llm
if "retriever" not in st.session_state:
    st.session_state["retriever"], st.session_state["llm"] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input
question = st.text_input("`Ваш вопрос:`")

if question:

    # Generate answer (w/ citations)
    import logging

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Write answer and sources
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain(
        {"question": question}, callbacks=[retrieval_streamer_cb, stream_handler]
    )
    answer.info("`Answer:`\n\n" + result["answer"])
    st.info("`Sources:`\n\n" + result["sources"])
