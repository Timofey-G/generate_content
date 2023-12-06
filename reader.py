from typing import List, Optional

from llama_index import download_loader
from llama_index.readers.schema.base import Document
import logging

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs.log")
formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger().setLevel(logging.WARNING)


class BeautifulSoupWebReader(download_loader("BeautifulSoupWebReader")):
    def load_data(
        self,
        urls: List[str],
        custom_hostname: Optional[str] = None,
        include_url_in_text: Optional[bool] = True,
    ) -> List[Document]:
        """Load data from the urls.

        Args:
            urls (List[str]): List of URLs to scrape.
            custom_hostname (Optional[str]): Force a certain hostname in the case
                a website is displayed under custom URLs (e.g. Substack blogs)
            include_url_in_text (Optional[bool]): Include the reference url in the text of the document

        Returns:
            List[Document]: List of documents.

        """
        from urllib.parse import urlparse

        import requests
        from bs4 import BeautifulSoup

        documents = []
        for url in urls:
            try:
                page = requests.get(url)
            except Exception:
                # raise ValueError(f"One of the inputs is not a valid url: {url}")
                logger.error(f"Не удалось получиться доступ к url: {url}")
                continue

            hostname = custom_hostname or urlparse(url).hostname or ""

            soup = BeautifulSoup(page.content, "html.parser")

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
                    line.strip() for line in soup.get_text(separator="\n").splitlines()
                ]
                data = "\n".join(line for line in lines if line)

            documents.append(Document(text=data, extra_info=extra_info))

        return documents
