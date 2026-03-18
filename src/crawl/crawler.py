import requests
import trafilatura
import json

urls = [
    "https://www.famousscientists.org/luis-alvarez/",
    "https://biographyonline.net/scientists/aristotle.html",
    "https://www.bbc.co.uk/teach/articles/zh8792p",
    "https://www.britannica.com/biography/Galileo-Galilei/Galileos-Copernicanism",
]

OUTPUT_FILE = "crawler_output.jsonl"
MIN_WORDS = 500


def extract_text_from_url(url):
    response = requests.get(url, timeout=15)
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
                print("Aucun texte extrait")
                continue

            wc = word_count(text)
            print(f"Nombre de mots: {wc}")

            if wc < MIN_WORDS:
                print("Page ignorée (trop courte)")
                continue

            data = {
                "url": url,
                "word_count": wc,
                "text": text
            }

            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept_pages += 1

            print("Page sauvegardée")

    print("\n================ RÉSUMÉ ================")
    print(f"Pages sauvegardées: {kept_pages}")
