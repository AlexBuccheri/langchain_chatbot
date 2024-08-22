"""

"""
import pickle

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time


def is_valid_url(url, base_url):
    # Check if the URL is valid and within the same domain
    parsed = urlparse(url)
    return bool(parsed.netloc) and parsed.netloc == urlparse(base_url).netloc


def get_all_links(url, base_url):
    # Fetch the webpage content and extract all the links
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag['href']
            # Build absolute URLs and check if they are internal links
            full_url = urljoin(base_url, href)
            if is_valid_url(full_url, base_url):
                links.add(full_url)
        return links
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return set()


def crawl_website(start_url):
    # Crawl the website starting from the start_url
    base_url = start_url
    visited = set()
    to_visit = {start_url}

    i = 0
    while to_visit:
        url = to_visit.pop()
        if url not in visited:
            visited.add(url)
            print(f"Crawling: {url}")
            links = get_all_links(url, base_url)
            to_visit.update(links - visited)
            time.sleep(0.1)  # Be polite to the server by adding a delay
        i += 1

        # Periodically dump visited sites to file
        if i % 60 == 0:
            print(f'Dumped to file after crawling {i} pages')
            with open('oct_webpages.txt', 'wb') as fid:
                pickle.dump(visited, fid)

    return visited


if __name__ == "__main__":
    start_url = "https://octopus-code.org/documentation/14/"

    with open('oct_webpages.txt', 'w') as f:
        pass

    all_pages = crawl_website(start_url)

    print("\nAll found pages:")

    with open('oct_webpages.txt', 'wb') as fid:
        pickle.dump(all_pages, fid)

    # for page in sorted(all_pages):
    #     print(page)
