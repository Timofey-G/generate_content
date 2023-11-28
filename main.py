import os
import argparse
import logging
from dotenv import load_dotenv
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
from reader import BeautifulSoupWebReader

load_dotenv()

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs.log")
formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

# logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOADER = BeautifulSoupWebReader()
INDEX_DIR = "storage"
SERVICE_CONTEXT = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-16k"))


def generate_article(urls, query):
    logger.info(f"Запрос: Query={query}, URLS={urls}")

    documents = LOADER.load_data(urls=urls)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    query_engine = index.as_query_engine(
        service_context=SERVICE_CONTEXT,
        similarity_top_k=10,
    )
    response = query_engine.query(
        f"""Напиши полноценную статью на тему: "{query}".
Напиши подробную и большую статью, раскрой все указанные темы полностью. Используй не меньше 1000 слов."""
    )

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
