import os
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from lxml import etree
from datasets import Dataset
from huggingface_hub import HfApi, login
from tqdm import tqdm

# Start from the first available adopted text page and follow "Volgende" links
START_TOC_URL = "https://www.europarl.europa.eu/doceo/document/TA-5-1999-07-21-TOC_NL.html"
HF_USERNAME = os.environ.get("HF_USERNAME", "vGassen")
HF_DATASET_NAME = "Dutch-European-Parliament-Adopted-Texts"
HF_REPO_ID = f"{HF_USERNAME}/{HF_DATASET_NAME}"


def collect_report_urls(start_url: str):
    urls = []
    visited = set()
    current = start_url

    session = requests.Session()
    while current and current not in visited:
        visited.add(current)
        resp = session.get(current, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        report_url = current.replace("-TOC_NL.html", "_NL.html")
        urls.append(report_url)
        next_link = soup.find("a", title="Volgende")
        if not next_link:
            next_link = soup.find("a", string=re.compile("Volgende", re.I))
        if not next_link or not next_link.get("href"):
            break
        current = urljoin(current, next_link["href"])
    return urls


def clean_text(text: str) -> str:
    """Apply common cleanup rules."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(The sitting (?:was suspended|opened|closed|ended) at.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Voting time ended at.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:debat|stemming|vraag|interventie)\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Het woord wordt gevoerd door:.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\(|\[)\s*(?:(?:[a-zA-Z]{2,3})\s*(?:|\s|))?\s*(?:artikel|rule|punt|item)\s*\d+(?:,\s*lid\s*\d+)?\s*(?:\s+\w+)?\s*(\)|\])", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(COM|A)\d+-\d+(/\d+)?\]", "", text)
    text = re.sub(r"\(?(?:http|https):\/\/[^\s]+?\)", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(COD\)\]", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(INI\)\]", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(RSP\)\]", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(IMM\)\]", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(NLE\)\]", "", text)
    text = re.sub(r"\[\s*\d{5}/\d{4}\s*-\s*C\d+-\d+/\d+\s*-\s*\d{4}/\d{4}\(NLE\)\]", "", text)
    text = re.sub(r"\(\u201cStemmingsuitslagen\u201d, punt \d+\)", "", text)
    text = re.sub(r"\(de Voorzitter(?: maakt na de toespraak van.*?| weigert in te gaan op.*?| stemt toe| herinnert eraan dat de gedragsregels moeten worden nageleefd| neemt er akte van|)\)", "", text)
    text = re.sub(r"\(zie bijlage.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*De vergadering wordt om.*?geschorst\.\)", "", text)
    text = re.sub(r"\(\s*De vergadering wordt om.*?hervat\.\)", "", text)
    text = re.sub(r"Volgens de \u201ccatch the eye\u201d-procedure wordt het woord gevoerd door.*?\.", "", text)
    text = re.sub(r"Het woord wordt gevoerd door .*?\.", "", text)
    text = re.sub(r"De vergadering wordt om \d{1,2}\.\d{2} uur gesloten.", "", text)
    text = re.sub(r"De vergadering wordt om \d{1,2}\.\d{2} uur geopend.", "", text)
    text = re.sub(r"Het debat wordt gesloten.", "", text)
    text = re.sub(r"Stemming:.*?\.", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def extract_dutch_text_from_xml(xml_content: bytes) -> str | None:
    """Parse XML verbatim report and return cleaned Dutch text."""
    try:
        parser = etree.XMLParser(recover=True, ns_clean=True)
        root = etree.fromstring(xml_content, parser=parser)
    except etree.XMLSyntaxError:
        return None

    dutch_nodes = root.xpath(
        '//*[translate(@xml:lang, "NL", "nl")="nl" or translate(@lang, "NL", "nl")="nl"]'
    )
    texts = []
    for node in dutch_nodes:
        text_content = "".join(node.itertext()).strip()
        if text_content:
            texts.append(text_content)

    if not texts:
        texts = [t.strip() for t in root.xpath("//text()") if t.strip()]

    final_text = clean_text("\n".join(texts))
    if final_text and len(final_text) > 50:
        return final_text
    return None


def extract_dutch_text_from_html(html_content: str) -> str | None:
    """Parse HTML verbatim report and return cleaned Dutch text."""
    soup = BeautifulSoup(html_content, "lxml")
    dutch_tags = [
        t for t in soup.find_all(True)
        if (t.get("lang", "").lower().startswith("nl") or t.get("xml:lang", "").lower().startswith("nl"))
    ]
    if dutch_tags:
        paragraphs = [t.get_text(" ", strip=True) for t in dutch_tags if t.get_text(strip=True)]
    else:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    final_text = clean_text("\n".join(paragraphs))
    if final_text and len(final_text) > 50:
        return final_text
    return None

def fetch_report_text(url: str, session: requests.Session) -> str | None:
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if url.endswith(".xml") or "xml" in content_type:
        return extract_dutch_text_from_xml(resp.content)
    return extract_dutch_text_from_html(resp.text)


def scrape() -> list:
    toc_urls = collect_report_urls(START_TOC_URL)
    data = []
    with requests.Session() as session:
        for url in tqdm(toc_urls, desc="Scraping reports"):
            try:
                text = fetch_report_text(url, session)
                if text:
                    data.append({"URL": url, "text": text, "source": "European Parliament Adopted Text"})
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
    return data


def push_dataset(records):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not provided")
        return
    login(token=token)
    ds = Dataset.from_list(records)
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    ds.push_to_hub(HF_REPO_ID, private=False)


def main():
    records = scrape()
    if records:
        push_dataset(records)
    else:
        print("No data scraped")


if __name__ == "__main__":
    main()
