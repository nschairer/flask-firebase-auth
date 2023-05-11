import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse, quote
import os

#function to replace invalid characters in the file name
def clean_filename(filename):
    invalid_chars = '<>:"/\\|?*&^%$#@!'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

# Define root domain to crawl
domain = "mydukaan.io"
full_url = "https://help.mydukaan.io"

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            
            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + quote(link)  # Use the quote function to handle special characters in the URL

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))

def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the JSON files
    if not os.path.exists("articles/"):
        os.mkdir("articles/")

    if not os.path.exists("articles/" + local_domain + "/"):
        os.mkdir("articles/" + local_domain + "/")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url)  # for debugging and to see the progress

        # Get the text from the URL using BeautifulSoup
        soup = BeautifulSoup(requests.get(url).text, "html.parser")

        # Extract headings and their corresponding text content
        articles = {}
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        for heading in headings:
            key = heading.get_text(strip=True)
            value = ""

            for elem in heading.next_siblings:
                if elem.name and elem.name.startswith('h'):
                    break
                if elem.string:
                    value += elem.string.strip()

            if key and value:
                articles[key] = value

        # Save the extracted articles as a JSON file
        file_name = 'articles/' + local_domain + '/' + clean_filename(url[8:].replace("/", "_")) + ".json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)
crawl(full_url)