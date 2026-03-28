import requests
import trafilatura
import json

urls = [
    # Alvarez
    "https://www.famousscientists.org/luis-alvarez/",
    "https://en.wikipedia.org/wiki/Luis_Walter_Alvarez",

    # Aristotle
    "https://biographyonline.net/scientists/aristotle.html",
    "https://en.wikipedia.org/wiki/Aristotle",

    # Galileo
    "https://www.britannica.com/biography/Galileo-Galilei",
    "https://en.wikipedia.org/wiki/Galileo_Galilei",

    # Newton
    "https://en.wikipedia.org/wiki/Isaac_Newton",
    "https://www.britannica.com/biography/Isaac-Newton",

    # Einstein
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://www.britannica.com/biography/Albert-Einstein",

    # Fermi
    "https://en.wikipedia.org/wiki/Enrico_Fermi",
    "https://www.britannica.com/biography/Enrico-Fermi",

    # Plato / Socrates
    "https://en.wikipedia.org/wiki/Plato",
    "https://en.wikipedia.org/wiki/Socrates",

    # Curie
    "https://en.wikipedia.org/wiki/Marie_Curie",
    "https://www.nobelprize.org/prizes/physics/1903/marie-curie/biographical/",

    # Darwin
    "https://en.wikipedia.org/wiki/Charles_Darwin",
    "https://www.britannica.com/biography/Charles-Darwin",

    # Copernicus
    "https://en.wikipedia.org/wiki/Nicolaus_Copernicus",
    "https://www.britannica.com/biography/Nicolaus-Copernicus",
]

OUTPUT_FILE = "data/crawler_output.jsonl"
MIN_WORDS = 500


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ScienceKGBot/1.0; student research project)"
}

def extract_text_from_url(url):
    response = requests.get(url, timeout=15, headers=HEADERS)
    html = response.text

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False
    )
    return text


def word_count(text):
    return len(text.split())


if __name__ == "__main__":
    kept_pages = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for url in urls:
            print("=" * 80)
            print(f"URL: {url}")

            text = extract_text_from_url(url)

            if text is None:
                print("No text extracted")
                continue

            wc = word_count(text)
            print(f"Word count: {wc}")

            if wc < MIN_WORDS:
                print("Page skipped (too short)")
                continue

            data = {
                "url": url,
                "word_count": wc,
                "text": text
            }

            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept_pages += 1

            print("Page saved")

    print("\n================ SUMMARY ================")
    print(f"Pages saved: {kept_pages}")