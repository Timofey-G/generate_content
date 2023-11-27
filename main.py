import os
import argparse
from dotenv import load_dotenv
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
from reader import BeautifulSoupWebReader

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOADER = BeautifulSoupWebReader()
INDEX_DIR = "storage"


def main(urls, query):
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo-16k")
    )

    # if not os.path.exists(INDEX_DIR):
    #     documents = LOADER.load_data(urls=urls)
    #     index = VectorStoreIndex.from_documents(documents)
    #     index.storage_context.persist(persist_dir=INDEX_DIR)
    # else:
    #     storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    #     index = load_index_from_storage(storage_context)

    documents = LOADER.load_data(urls=urls)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    query_engine = index.as_query_engine(
        service_context=service_context,
        similarity_top_k=10,
    )
    response = query_engine.query(
        f"""Напиши полноценную статью на тему: "{query}".
Напиши подробную и большую статью, раскрой все указанные темы полностью. Используй не меньше 1000 слов."""
    )
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
    result = main(urls=args.urls, query=args.query)
    print("-" * 100)
    print(result)
