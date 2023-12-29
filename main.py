import asyncio
import os
from typing import List, Tuple
import time

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
    summarization_statements: str
    verified_statements: str
    article_statements: str


def timing_decorator(func):
    async def wrapper(*args, **kwargs):
        article = args[0]
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        time_elapsed = end_time - start_time
        print(
            f"Функция {func.__name__} выполнялась {time_elapsed:.2f} секунд для статьи с URL: {article.url}"
        )
        return result

    return wrapper


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


async def summarize(topic: str, text: str, max_length: int) -> str:
    prompt = (
        "Перепишите следующий текст, сосредоточив внимание на удалении общей и очевидной информации, которая не добавляет новых знаний. "
        "Старайтесь исключить информацию, которая не приносит практической пользы или новых знаний. "
        f"Также игнорируй информацию, которая не относится к теме '{topic}'. "
        "Обязательно сохраните все конкретные утверждения, факты, цифры, инструкции, рецепты и т.д. "
        "Текст должен сохранить все детали, такие как специфические рекомендации и примеры описываемых вещей, "
        "удаляй только незначимую информацию, а все конкретные примеры, характеристики и описание вещей оставь нетронутыми. "
        "Ответ должен быть на русском языке. "
        f"Вот текст для переработки: ```{text[:max_length]}```"
    )
    chat_completion = await async_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4-1106-preview",
        temperature=0.0,
    )
    return chat_completion.choices[0].message.content


@timing_decorator
async def process_article(article, max_length=20000):
    # article.summarized_text = ""
    # return
    try:
        article.summarized_text = await summarize(
            article.topic, article.text, max_length
        )
    except Exception as e:
        print("ОШИБКА", e)
        logger.error(f"Ошибка при суммаризации статьи: {article.url}, Ошибка: {e}")
        article.summarized_text = ""
        return


def answer(topic: str, articles: List[Article], separator="*" * 124) -> str:
    articles_info = f"\n\n{separator[:len(separator)]}\n\n".join(
        [
            f"{art.url}\n{art.summarized_text}\n\n{separator[:len(separator)//4]}\n"
            f"Необоснованные утверждения в статье:\n{art.article_statements}"
            for art in articles
        ]
    )

    pattern = f"""Ключевая информация, полученная с сайтов:
{articles_info}"""

    return pattern


async def text_analysis(topic, text, max_length) -> str:
    prompt = (
        f"Я предоставлю тебе текст, твоя задача проанализировать его, и выделить сомнительные утверждения в тексте относящиеся к теме: '{topic}', "
        "которые совсем не похожи на правду и никак не обоснованы. "
        f"Все такие утверждения выдели в ответе в формате нумерованного списка. Вот текст: ```{text[:max_length]}```."
    )
    chat_completion = await async_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo-1106",
        temperature=0.0,
    )
    return chat_completion.choices[0].message.content


@timing_decorator
async def article_analysis(article, max_length=20000):
    try:
        article.article_statements = await text_analysis(
            article.topic, article.text, max_length
        )
    except Exception as e:
        print("ОШИБКА", e)
        logger.error(f"Ошибка при анализе статьи: {article.url}, Ошибка: {e}")
        article.article_statements = ""
        return


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
        articles.append(article)

    logger.info(f"Количество успешных извлеченных статей {len(articles)}")

    requests_list = [process_article(article) for article in articles]
    requests_list.extend([article_analysis(article) for article in articles])
    await asyncio.gather(*requests_list)

    return answer(topic, articles)
