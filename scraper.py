import os
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from lxml import etree
from datasets import Dataset
from huggingface_hub import HfApi, login
from tqdm import tqdm

# Start from the first available minutes page and follow "Volgende" links
START_TOC_URL = "https://www.europarl.europa.eu/doceo/document/PV-5-2003-05-12-TOC_NL.html"
HF_USERNAME = os.environ.get("HF_USERNAME", "YOUR_HUGGINGFACE_USERNAME")
HF_DATASET_NAME = "Dutch-European-Parliament-Minutes"
HF_REPO_ID = f"{HF_USERNAME}/{HF_DATASET_NAME}"

NAMESPACES = {
    "text": "http://openoffice.org/2000/text",
    "table": "http://openoffice.org/2000/table",
}


def collect_minutes_urls(start_url: str):
    urls = []
    visited = set()
    current = start_url

    session = requests.Session()
    while current and current not in visited:
        visited.add(current)
        resp = session.get(current, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        minutes_url = current.replace("-TOC_NL.html", "_NL.xml")
        urls.append(minutes_url)
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
    """Parse XML minutes and return cleaned Dutch text."""
    try:
        parser = etree.XMLParser(recover=True, ns_clean=True)
        root = etree.fromstring(xml_content, parser=parser)
    except etree.XMLSyntaxError:
        return None

    dutch_texts = []
    relevant_sections = [
        "PV.Other.Text",
        "PV.Debate.Text",
        "PV.Vote.Text",
        "PV.Sitting.Resumption.Text",
        "PV.Approval.Text",
        "PV.Agenda.Text",
        "PV.Sitting.Closure.Text",
    ]

    for section in relevant_sections:
        xpath_query = f"//{section}//text:p"
        for p_tag in root.xpath(xpath_query, namespaces=NAMESPACES):
            text_content = p_tag.xpath("string()").strip()
            if not text_content:
                continue
            if p_tag.xpath("ancestor::table:table", namespaces=NAMESPACES):
                continue
            if (
                p_tag.xpath("./Orator.List.Text", namespaces=NAMESPACES)
                or p_tag.xpath("./Attendance.Participant.Name", namespaces=NAMESPACES)
            ):
                name_list_text = p_tag.xpath(
                    "string(./Orator.List.Text)", namespaces=NAMESPACES
                ).strip()
                if (
                    len(text_content) < 100
                    and name_list_text
                    and name_list_text == text_content
                ):
                    continue
            if len(text_content) < 20 and not re.search(r"[a-zA-Z]{5,}", text_content):
                continue
            dutch_texts.append(text_content)

    final_text = clean_text("\n".join(dutch_texts))
    if final_text and len(final_text) > 50:
        return final_text
    return None


def extract_dutch_text_from_html(html_content: str) -> str | None:
    """Parse HTML minutes and return cleaned Dutch text."""
    soup = BeautifulSoup(html_content, "lxml")
    paragraphs = [
        p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
    ]
    final_text = clean_text("\n".join(paragraphs))
    if final_text and len(final_text) > 50:
        return final_text
    return None


def fetch_minutes_text(url: str, session: requests.Session) -> str | None:
    resp = session.get(url, timeout=20)
    if resp.status_code == 404 and url.endswith("_NL.xml"):
        # Older minutes might only be available in HTML format
        html_url = url.replace("_NL.xml", "_NL.html")
        resp = session.get(html_url, timeout=20)
        resp.raise_for_status()
        return extract_dutch_text_from_html(resp.text)
    resp.raise_for_status()
    return extract_dutch_text_from_xml(resp.content)


def scrape() -> list:
    toc_urls = collect_minutes_urls(START_TOC_URL)
    data = []
    with requests.Session() as session:
        for url in tqdm(toc_urls, desc="Scraping minutes"):
            try:
                text = fetch_minutes_text(url, session)
                if text:
                    data.append({"URL": url, "text": text, "source": "European Parliament Minutes"})
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
