import os
import argparse
from dotenv import load_dotenv
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import LangChainLLM
from reader import BeautifulSoupWebReader, logger
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.callbacks.base import CallbackManager
from llama_index.logger import LlamaLogger
from langchain.chat_models.gigachat import GigaChat
import os
from llama_index import VectorStoreIndex
from llama_index.storage import StorageContext

# from llama_index.llms import OpenAI, LangChainLLM
from llama_index import LLMPredictor, ServiceContext

from llama_index.indices.postprocessor import (
    SimilarityPostprocessor,
    LongContextReorder,
)
from llama_index.indices.postprocessor import CohereRerank
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate
from llama_index.response_synthesizers import get_response_synthesizer
from langchain.chat_models.gigachat import GigaChat
from llama_index.indices.postprocessor import (
    SimilarityPostprocessor,
    LongContextReorder,
)
from llama_index.indices.postprocessor import CohereRerank


load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GIGA_CHAT_API_KEY = os.getenv("CREDENTIALS")
giga_chat = GigaChat(credentials=GIGA_CHAT_API_KEY, verify_ssl_certs=False)
LOADER = BeautifulSoupWebReader()
INDEX_DIR = "storage"


def get_query_engine(index):
    response_synthesizer = get_response_synthesizer(
        streaming=True,
        response_mode="tree_summarize",
        verbose=True,
        service_context=ServiceContext.from_defaults(llm=LangChainLLM(giga_chat)),
        summary_template=ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    content=(
                        "Напиши подробную и большую статью, раскрой все указанные темы полностью. Используй не меньше 1000 слов. Ответь на русском языке."
                    ),
                    role=MessageRole.SYSTEM,
                ),
                ChatMessage(
                    content=("Напиши полноценную статью на тему: '{query_str}'"),
                    role=MessageRole.USER,
                ),
            ]
        ),
    )

    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.82),
        # CohereRerank(top_n=1, api_key=COHERE_API_KEY),
        # MyPostprocessor2(),
        # LongContextReorder(),
    ]

    chat_engine = index.as_query_engine(
        streaming=True,
        chat_mode="context",
        response_synthesizer=response_synthesizer,
        similarity_top_k=10,
        # node_postprocessors=node_postprocessors,
    )
    return chat_engine


def generate_article(urls, query):
    logger.info(f"Запрос: Query={query}, URLS={urls}")

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    llama_logger = LlamaLogger()
    service_context = ServiceContext.from_defaults(
        llm=LangChainLLM(giga_chat),
        llama_logger=llama_logger,
        callback_manager=callback_manager,
    )

    documents = LOADER.load_data(urls=urls)
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
    )
    index.storage_context.persist(persist_dir=INDEX_DIR)

    query_engine = get_query_engine(index)
    response = query_engine.query(query)

    logger.info(f"Ответ: {response}")
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обработка статей и создание запросов")
    parser.add_argument(
        "-u",
        "--urls",
        nargs="+",
        required=True,
        help="Список URL-адресов для обработки",
    )
    parser.add_argument("-q", "--query", required=True, help="Тема статьи для запроса")

    args = parser.parse_args()
    result = generate_article(urls=args.urls, query=args.query)
    print("-" * 100)
    print(result)
