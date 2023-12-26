import asyncio
import os
from typing import List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

from reader import BeautifulSoupWebReader, logger

load_dotenv()
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LOADER = BeautifulSoupWebReader()


class Article:
    topic: str
    url: str
    full_text: str
    text: str
    summarized_text: str
    density: List[int]
    start: int
    end: int


def chart(text: str) -> List[int]:
    rows = text.split("\n")
    density = []
    for row in rows:
        density.append(len(row.split()))
    return density


def extract_article(text, start, end):
    lines = text.split("\n")
    if start is not None and end is not None and 0 <= start < end <= len(lines):
        return "\n".join(lines[start:end])
    else:
        return ""


def extract_span(
    density_list: List[int],
    threshold=7,
    min_run=3,
    min_below_threshold_run=10,
    string_length=30,
) -> Tuple[int]:
    start = None
    end = len(density_list)
    current_run = 0
    current_below_threshold_run = 0
    is_text_segment = False

    for i, density in enumerate(density_list):
        if density > threshold:
            current_below_threshold_run = 0
            current_run += 1
            if start is None and (current_run >= min_run or density > string_length):
                if i > min_run:
                    start = i - min_run + 1
                else:
                    start = i
                is_text_segment = True
            if current_run >= min_run:
                is_text_segment = True
        else:
            current_below_threshold_run += 1
            if (
                start is not None
                and current_below_threshold_run >= min_below_threshold_run
                and is_text_segment is True
            ):
                end = i - current_below_threshold_run + 1
                is_text_segment = False
            current_run = 0

    return start, end


# async def summarize(topic: str, text: str, text_length: int, answer_length=2000) -> str:
#     prompt = (
#         f"Your task is to create a more concise version of the content related to the topic '{topic}'. "
#         "It is important to identify and focus on the main information related to this topic, "
#         "while ensuring that all key facts and instructions are preserved in their complete form. "
#         "Please ignore any unrelated content such as comments, side news, or information from headers or sidebars. "
#         "The response should be in Russian and should provide a simplified yet detailed presentation, "
#         f"ensuring that no critical details are omitted and the text contains at least {answer_length} characters. "
#         f"Here is the text to be reworked: ```{text[:text_length]}```"
#     )
#     chat_completion = await async_client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model="gpt-3.5-turbo-16k",
#     )
#     return chat_completion.choices[0].message.content


async def summarize(topic: str, text: str, text_length: int, answer_length=2000) -> str:
    # print("*" * 150)
    prompt = (
        f"Перепишите следующий текст, убрав всю очевидную и общую информацию. "
        "Обязательно сохраните конкретные утверждения, факты, цифры, инструкции, рецепты и т.д. "
        f"Ответ должен быть на русском языке, все важные данные и факты должны остаться нетронутыми. "
        f"Вот текст для переработки: ```{text[:text_length]}```"
    )
    # print(prompt)
    chat_completion = await async_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4-1106-preview",
    )
    # print("*" * 150)
    # print(chat_completion.choices[0].message.content)

    # input_word_count = count_words(prompt)
    # output_word_count = count_words(chat_completion.choices[0].message.content)

    # metrics = load_metrics()

    # metrics["input_words"] += input_word_count
    # metrics["output_words"] += output_word_count
    # save_metrics(metrics)

    # print(f"Текущее общее количество входящих слов: {metrics['input_words']}")
    # print(f"Текущее общее количество исходящих слов: {metrics['output_words']}")
    # print(
    #     f"Текущее общее количество баксов входящих: {round(((metrics['input_words']/750)*1000)/100000,2)}"
    # )
    # print(
    #     f"Текущее общее количество баксов исходящих: {round(((metrics['output_words']/750)*1000)/100000,2)}"
    # )
    # print(
    #     f"Текущее общее количество баксов всего: {round((((metrics['output_words']/750)*1000)/100000)+(((metrics['input_words']/750)*1000)/100000),2)}"
    # )

    return chat_completion.choices[0].message.content


async def process_article(article, max_length=20000):
    try:
        article.summarized_text = await summarize(
            article.topic, article.text, max_length
        )
    except Exception as e:
        print("ОШИБКА", e)
        logger.error(f"Ошибка при обработке статьи: {article.url}, Ошибка: {e}")
        article.summarized_text = ""


def answer(topic: str, articles: List[Article], separator="*" * 124) -> str:
    articles_info = f"\n\n{separator[:len(separator)]}\n\n".join(
        [f"{art.url}\n{art.summarized_text}" for art in articles]
    )

    pattern = f"""Ключевая информация, полученная с сайтов:
{articles_info}"""

    return pattern


async def main(urls, topic):
    logger.info(f"ЗАПРОС: {topic}")
    logger.info(f"Количество полученных url: {len(urls)}")
    data = await LOADER.load_data(urls=urls)
    logger.info(f"Количество спарсенных сайтов: {len(data)}")

    articles = []
    for elem in data:
        article = Article()
        article.topic = topic
        article.url = elem.metadata["URL"]
        article.full_text = elem.text
        if (
            len(article.full_text) < 400
            or "forbidden" in article.full_text.lower()
            or "blocked" in article.full_text.lower()
        ):
            logger.error(
                f"Не удалось получить данные с url: {article.url}, Текст ответа: {article.full_text}"
            )
            continue
        article.density = chart(article.full_text)
        article.start, article.end = extract_span(article.density)
        article.text = extract_article(article.full_text, article.start, article.end)
        print(elem.text)
        articles.append(article)

    logger.info(f"Количество успешных извлеченных статей {len(articles)}")
    await asyncio.gather(*[process_article(article) for article in articles])

    context = ""
    for article in articles:
        context += article.summarized_text + "\n\n\n"
    context = context[:-3]

    return answer(topic, articles)


# import json

# log_file = "tokens.json"


# def count_words(text: str) -> int:
#     """Подсчитывает количество слов в предоставленном тексте."""
#     words = text.split()
#     return len(words)


# def load_metrics():
#     """Загружает текущие метрики из файла."""
#     try:
#         with open(log_file, "r") as file:
#             return json.load(file)
#     except (FileNotFoundError, json.JSONDecodeError):
#         return {"input_words": 0, "output_words": 0}


# def save_metrics(metrics):
#     """Сохраняет метрики в файл."""
#     with open(log_file, "w") as file:
#         json.dump(metrics, file)
