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
    threshold=5,
    min_run=3,
    min_below_threshold_run=10,
    string_length=20,
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


async def summarize(topic: str, text: str, text_length: int, answer_length=2000) -> str:
    prompt = (
        f"Your task is to create a more concise version of the content related to the topic '{topic}'. "
        "It is important to identify and focus on the main information related to this topic, "
        "while ensuring that all key facts and instructions are preserved in their complete form. "
        "Please ignore any unrelated content such as comments, side news, or information from headers or sidebars. "
        "The response should be in Russian and should provide a simplified yet detailed presentation, "
        f"ensuring that no critical details are omitted and the text contains at least {answer_length} characters. "
        f"Here is the text to be reworked: ```{text[:text_length]}```"
    )
    chat_completion = await async_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo-16k",
    )
    return chat_completion.choices[0].message.content


async def generate_article(topic: str, context: str, article_length=10000) -> str:
    prompt = (
        f"Task: Write a comprehensive and detailed article on the topic '{topic}'. "
        "Use the information provided below, collected from various sources. Each source is separated by three line breaks. "
        "The article should be well-structured, engaging, and informative, covering all important aspects of the topic. "
        "Please focus on incorporating as many specific facts and examples as possible from the provided information. "
        "The content should reflect the details and data in the summaries, making the article factually rich and precise. "
        "The content should be original and not a direct copy of the provided summaries. "
        f"The desired length of the article is approximately {article_length} characters. "
        "The response should be in Russian. "
        f"Here is information to use: ```\n{context}\n```"
    )
    chat_completion = await async_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo-16k",
    )
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


def answer(topic: str, article: str, articles: List[Article], separator="*") -> str:
    articles_info = f"\n\n{separator*124}\n\n".join(
        [f"{art.url}\n{art.summarized_text}" for art in articles]
    )

    pattern = f"""Статья: {topic}
{article}


{separator*248}


Ключевая информация, полученная с сайтов:
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
        articles.append(article)

    logger.info(f"Количество успешных извлеченных статей {len(articles)}")
    await asyncio.gather(*[process_article(article) for article in articles])

    context = ""
    for article in articles:
        context += article.summarized_text + "\n\n\n"

    article = await generate_article(topic, context)
    return answer(topic, article, articles)
