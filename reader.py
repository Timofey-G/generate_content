import asyncio
import logging
from typing import List, Optional
from urllib.parse import urlparse

import aiohttp
import requests
from bs4 import BeautifulSoup
from llama_index import download_loader
from llama_index.readers.schema.base import Document

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/logs.log")
formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger().setLevel(logging.WARNING)


class BeautifulSoupWebReader(download_loader("BeautifulSoupWebReader")):
    async def fetch(self, session, url, headers):
        try:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientResponseError as e:
            logger.error(f"Ошибка HTTP при доступе к url: {url}, ошибка: {e}")
        except asyncio.TimeoutError:
            logger.error(f"Время ожидания ответа от {url} истекло.")
        except Exception as e:
            logger.error(f"Ошибка при доступе к url: {url}, ошибка: {e}")

    async def load_data(
        self,
        urls: List[str],
        custom_hostname: Optional[str] = None,
        include_url_in_text: Optional[bool] = True,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
        timeout: int = 30,
    ) -> List[Document]:
        headers = {"User-Agent": user_agent}
        documents = []

        timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [self.fetch(session, url, headers) for url in urls]
            pages = await asyncio.gather(*tasks)

            for page, url in zip(pages, urls):
                if page is None:
                    continue

                hostname = custom_hostname or urlparse(url).hostname or ""
                soup = BeautifulSoup(page, "html.parser")
                data = ""
                extra_info = {"URL": url}

                if hostname in self.website_extractor:
                    data, metadata = self.website_extractor[hostname](
                        soup=soup, url=url, include_url_in_text=include_url_in_text
                    )
                    extra_info.update(metadata)

                else:
                    for script_or_style in soup(["script", "style"]):
                        script_or_style.decompose()

                    lines = [
                        line.strip()
                        for line in soup.get_text(separator="\n").splitlines()
                    ]
                    data = "\n".join(line for line in lines if line)

                documents.append(Document(text=data, extra_info=extra_info))

        return documents
