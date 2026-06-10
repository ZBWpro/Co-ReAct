import json
import os
import time
from typing import Dict, Optional

import dotenv
import requests
from typing_extensions import TypedDict

from ..cache import cached

dotenv.load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
CLOUDSWAY_API_KEY = os.getenv("CLOUDSWAY_API_KEY")
CLOUDSWAY_READER_URL = os.getenv("CLOUDSWAY_READER_URL", "")
TIMEOUT = int(os.getenv("API_TIMEOUT", 30))
MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", 1))
RETRY_BACKOFF = float(os.getenv("API_RETRY_BACKOFF", 2.0))


class JinaMetadata(TypedDict, total=False):
    lang: str
    viewport: str


class JinaWebpageResponse(TypedDict, total=False):
    url: str
    title: str
    content: str
    description: str
    publishedTime: str
    metadata: JinaMetadata
    success: bool
    error: str


@cached()
def fetch_webpage_content_jina(
    url: str,
    api_key: str = None,
    timeout: int = TIMEOUT,
) -> JinaWebpageResponse:
    """
    Fetch webpage content using Cloudsway Reader API (compatible with Jina Reader format).

    Args:
        url: The URL of the webpage to fetch
        api_key: Jina API key (if not provided, will use CLOUDSWAY_API_KEY or JINA_API_KEY env var)
        timeout: Request timeout in seconds (if not provided, will use TIMEOUT env var or default 30)

    Returns:
        JinaWebpageResponse containing:
        - url: The original URL that was fetched
        - title: The webpage title
        - content: The webpage content as clean text/markdown
        - description: The webpage description (if available)
        - publishedTime: Publication timestamp (if available)
        - metadata: Additional metadata (lang, viewport, etc.)
        - success: Boolean indicating if the fetch was successful
        - error: Error message if fetch failed
    """
    cloudsway_key = os.getenv("CLOUDSWAY_API_KEY")
    if cloudsway_key:
        return _fetch_webpage_cloudsway_jina(url, cloudsway_key, timeout)

    if not api_key:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError(
                "Neither CLOUDSWAY_API_KEY nor JINA_API_KEY environment variable is set"
            )

    if timeout is None:
        timeout = TIMEOUT

    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    response = requests.get(jina_url, headers=headers, timeout=timeout)

    if response.status_code != 200:
        raise Exception(
            f"API request failed with status {response.status_code}: {response.text}"
        )

    json_response = response.json()

    # Extract data from JSON response
    data = json_response.get("data", {})

    return {
        "url": data.get("url", url),
        "title": data.get("title", ""),
        "content": data.get("content", ""),
        "description": data.get("description", ""),
        "publishedTime": data.get("publishedTime", ""),
        "metadata": data.get("metadata", {}),
        "success": True,
    }


def _fetch_webpage_cloudsway_jina(
    url: str,
    cloudsway_key: str,
    timeout: int = TIMEOUT,
) -> JinaWebpageResponse:
    """Fetch webpage content using Cloudsway Reader API and convert to Jina format."""
    if timeout is None:
        timeout = TIMEOUT

    headers = {
        "Authorization": f"Bearer {cloudsway_key}",
        "Content-Type": "application/json",
    }

    payload = json.dumps({
        "url": url,
        "formats": ["TEXT"],
        "mode": "auto",
        "timeout": 10 * 1000,
    }, ensure_ascii=False)

    try:
        last_err = None
        response = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = requests.post(
                    CLOUDSWAY_READER_URL,
                    headers=headers,
                    data=payload,
                    timeout=timeout,
                )
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue

        if response is None:
            return {
                "url": url,
                "title": "",
                "content": "",
                "description": "",
                "publishedTime": "",
                "metadata": {},
                "success": False,
                "error": f"Cloudsway reader failed after {1 + MAX_RETRIES} attempts: {last_err}",
            }

        if response.status_code != 200:
            return {
                "url": url,
                "title": "",
                "content": "",
                "description": "",
                "publishedTime": "",
                "metadata": {},
                "success": False,
                "error": f"Cloudsway reader API failed with status {response.status_code}: {response.text}",
            }

        data = response.json()
        text_content = data.get("text", "")

        # Extract a title from the first line of text content if available
        title = ""
        if text_content:
            first_line = text_content.split("\n")[0].strip()
            if len(first_line) < 200:
                title = first_line

        return {
            "url": url,
            "title": title,
            "content": text_content,
            "description": "",
            "publishedTime": "",
            "metadata": data.get("usage", {}),
            "success": True,
        }

    except requests.exceptions.RequestException as e:
        return {
            "url": url,
            "title": "",
            "content": "",
            "description": "",
            "publishedTime": "",
            "metadata": {},
            "success": False,
            "error": f"Error fetching webpage via Cloudsway: {str(e)}",
        }
