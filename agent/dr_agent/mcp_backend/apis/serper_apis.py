import json
import os
import time
import urllib.parse
from typing import Dict, List, Optional, Union

import dotenv
import requests
from typing_extensions import TypedDict

from ..cache import cached

MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", 3))
RETRY_BACKOFF = float(os.getenv("API_RETRY_BACKOFF", 2.0))

# Load environment variables
dotenv.load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
CLOUDSWAY_API_KEY = os.getenv("CLOUDSWAY_API_KEY")
CLOUDSWAY_SEARCH_URL = os.getenv("CLOUDSWAY_SEARCH_URL", "")
CLOUDSWAY_SCHOLAR_URL = os.getenv("CLOUDSWAY_SCHOLAR_URL", "")
CLOUDSWAY_READER_URL = os.getenv("CLOUDSWAY_READER_URL", "")
TIMEOUT = int(os.getenv("API_TIMEOUT", 10))


class KnowledgeGraph(TypedDict, total=False):
    title: str
    type: str
    website: str
    imageUrl: str
    description: str
    descriptionSource: str
    descriptionLink: str
    attributes: Optional[Dict[str, str]]


class Sitelink(TypedDict):
    title: str
    link: str


class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[List[Sitelink]]
    attributes: Optional[Dict[str, str]]
    date: Optional[str]


class PeopleAlsoAsk(TypedDict):
    question: str
    snippet: str
    title: str
    link: str


class RelatedSearch(TypedDict):
    query: str


class SearchResponse(TypedDict, total=False):
    searchParameters: Dict[str, Union[str, int, bool]]
    knowledgeGraph: Optional[KnowledgeGraph]
    organic: List[SearchResult]
    peopleAlsoAsk: Optional[List[PeopleAlsoAsk]]
    relatedSearches: Optional[List[RelatedSearch]]


class ScholarResult(TypedDict):
    title: str
    link: str
    publicationInfo: str
    snippet: str
    year: Union[int, str]
    citedBy: int


class ScholarResponse(TypedDict):
    searchParameters: Dict[str, Union[str, int, bool]]
    organic: List[ScholarResult]


class WebpageContentResponse(TypedDict, total=False):
    url: str
    text: str
    markdown: str
    metadata: Dict[str, Union[str, int, bool]]
    credits: int


@cached()
def search_serper(
    query: str,
    num_results: int = 10,
    gl: str = "us",
    hl: str = "en",
    search_type: str = "search",  # Can be "search", "places", "news", "images"
    api_key: str = None,
) -> SearchResponse:
    """
    Search using Cloudsway API (compatible with Serper.dev format) for general web search.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
        gl: Country code to boosts search results whose country of origin matches the parameter value (default: us)
        hl: Host language of user interface (default: en)
        search_type: Type of search to perform (default: "search")
                    Options: "search", "places", "news", "images"
        api_key: Serper API key (if not provided, will use CLOUDSWAY_API_KEY or SERPER_API_KEY env var)

    Returns:
        SearchResponse containing:
        - searchParameters: Dict with search metadata
        - knowledgeGraph: Optional knowledge graph information
        - organic: List of organic search results
        - peopleAlsoAsk: Optional list of related questions
        - relatedSearches: Optional list of related search queries
    """
    # Guard against empty queries
    if not query or not query.strip():
        return SearchResponse(
            searchParameters={"q": query, "error": "empty query"},
            organic=[],
        )

    cloudsway_key = os.getenv("CLOUDSWAY_API_KEY")
    if cloudsway_key:
        return _search_cloudsway(query, num_results)

    if not api_key:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "Neither CLOUDSWAY_API_KEY nor SERPER_API_KEY environment variable is set"
            )

    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query, "num": num_results, "gl": gl, "hl": hl, "type": search_type})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=payload)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper search: {str(e)}")


def _search_cloudsway(query: str, num_results: int = 10) -> SearchResponse:
    """Search using Cloudsway API and convert response to Serper format."""
    cloudsway_key = os.getenv("CLOUDSWAY_API_KEY")
    headers = {"Authorization": f"Bearer {cloudsway_key}"}
    # Decode in case the model generated URL-encoded query text (e.g. %27 instead of ')
    query = urllib.parse.unquote(query)

    try:
        last_err = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = requests.get(
                    CLOUDSWAY_SEARCH_URL,
                    params={"q": query},
                    headers=headers,
                    timeout=TIMEOUT,
                )

                if response.status_code != 200:
                    raise Exception(
                        f"Cloudsway search API failed with status {response.status_code}: {response.text}"
                    )

                data = response.json()
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF + attempt)
                    continue
                raise Exception(f"Cloudsway search failed after {1 + MAX_RETRIES} attempts: {e}")
        else:
            raise Exception(f"Cloudsway search failed: {last_err}")

        web_pages = data.get("webPages", {}).get("value", [])

        organic_results = []
        for position, item in enumerate(web_pages[:num_results], start=1):
            organic_results.append({
                "title": item.get("name", ""),
                "link": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "position": position,
            })

        return {
            "searchParameters": {
                "q": query,
                "num": num_results,
                "type": "search",
            },
            "organic": organic_results,
        }

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Cloudsway search: {str(e)}")


@cached()
def search_serper_scholar(
    query: str,
    num_results: int = 10,
    api_key: str = None,
) -> ScholarResponse:
    """
    Search academic papers using Cloudsway API (compatible with Serper.dev Scholar format).

    Args:
        query: Academic search query string
        num_results: Number of results to return (default: 10)
        api_key: Serper API key (if not provided, will use CLOUDSWAY_API_KEY or SERPER_API_KEY env var)

    Returns:
        ScholarResponse containing:
        - organic: List of academic paper results with:
            - title: Paper title
            - link: URL to the paper
            - publicationInfo: Author and publication details
            - snippet: Brief excerpt from the paper
            - year: Publication year
            - citedBy: Number of citations
    """
    cloudsway_key = os.getenv("CLOUDSWAY_API_KEY")
    if cloudsway_key:
        return _search_cloudsway_scholar(query, num_results)

    if not api_key:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "Neither CLOUDSWAY_API_KEY nor SERPER_API_KEY environment variable is set"
            )

    url = "https://google.serper.dev/scholar"

    payload = json.dumps({"q": query, "num": num_results})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=payload)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper scholar search: {str(e)}")


def _search_cloudsway_scholar(query: str, num_results: int = 10) -> ScholarResponse:
    """Search academic papers using Cloudsway API and convert to Serper Scholar format."""
    cloudsway_key = os.getenv("CLOUDSWAY_API_KEY")
    headers = {"Authorization": f"Bearer {cloudsway_key}"}
    # Decode in case the model generated URL-encoded query text
    query = urllib.parse.unquote(query)

    try:
        last_err = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = requests.get(
                    CLOUDSWAY_SCHOLAR_URL,
                    params={"q": query},
                    headers=headers,
                    timeout=TIMEOUT,
                )

                if response.status_code != 200:
                    raise Exception(
                        f"Cloudsway scholar API failed with status {response.status_code}: {response.text}"
                    )

                data = response.json()
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF + attempt)
                    continue
                raise Exception(f"Cloudsway scholar search failed after {1 + MAX_RETRIES} attempts: {e}")
        else:
            raise Exception(f"Cloudsway scholar search failed: {last_err}")

        web_pages = data.get("webPages", {}).get("value", [])

        organic_results = []
        for item in web_pages[:num_results]:
            cited_by_raw = item.get("citeBy", "0")
            try:
                cited_by = int(cited_by_raw)
            except (ValueError, TypeError):
                cited_by = 0

            organic_results.append({
                "title": item.get("name", ""),
                "link": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "publicationInfo": item.get("author", ""),
                "year": item.get("datePublished", ""),
                "citedBy": cited_by,
            })

        return {
            "searchParameters": {
                "q": query,
                "num": num_results,
                "type": "scholar",
            },
            "organic": organic_results,
        }

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Cloudsway scholar search: {str(e)}")


@cached()
def fetch_webpage_content(
    url: str,
    include_markdown: bool = True,
    api_key: str = None,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Cloudsway Reader API (compatible with Serper.dev format).

    Args:
        url: The URL of the webpage to fetch
        include_markdown: Whether to include markdown formatting in the response (default: True)
        api_key: Serper API key (if not provided, will use CLOUDSWAY_API_KEY or SERPER_API_KEY env var)

    Returns:
        WebpageContentResponse containing:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
    """
    cloudsway_key = os.getenv("CLOUDSWAY_API_KEY")
    if cloudsway_key:
        return _fetch_webpage_cloudsway(url)

    if not api_key:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "Neither CLOUDSWAY_API_KEY nor SERPER_API_KEY environment variable is set"
            )

    scrape_url = "https://scrape.serper.dev"

    payload = json.dumps({"url": url, "includeMarkdown": include_markdown})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(scrape_url, headers=headers, data=payload)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        data["url"] = url
        return data

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching webpage content: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing API response: {str(e)}")


def _fetch_webpage_cloudsway(url: str) -> WebpageContentResponse:
    """Fetch webpage content using Cloudsway Reader API and convert to Serper format."""
    cloudsway_key = os.getenv("CLOUDSWAY_API_KEY")
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
                    timeout=30,
                )
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF + attempt)
                    continue

        if response is None:
            raise Exception(f"Cloudsway reader failed after {1 + MAX_RETRIES} attempts: {last_err}")

        if response.status_code != 200:
            raise Exception(
                f"Cloudsway reader API failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        text_content = data.get("text", "")

        return {
            "url": url,
            "text": text_content,
            "markdown": text_content,
            "metadata": data.get("usage", {}),
        }

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching webpage via Cloudsway: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing Cloudsway reader response: {str(e)}")


# Example usage:
if __name__ == "__main__":
    # Regular search example
    try:
        results = search_serper("apple inc", num_results=5)
        print("Regular Search Results:")
        print(f"Found {len(results.get('organic', []))} results")
        if "knowledgeGraph" in results:
            print(f"Knowledge Graph: {results['knowledgeGraph']['title']}")
        print()
    except Exception as e:
        print(f"Search error: {e}")

    # Scholar search example
    try:
        scholar_results = search_serper_scholar(
            "attention is all you need", num_results=5
        )
        print("Scholar Search Results:")
        print(f"Found {len(scholar_results.get('organic', []))} academic papers")
        for paper in scholar_results.get("organic", [])[:2]:
            print(
                f"- {paper['title']} ({paper['year']}) - Cited by: {paper['citedBy']}"
            )
        print()
    except Exception as e:
        print(f"Scholar search error: {e}")
