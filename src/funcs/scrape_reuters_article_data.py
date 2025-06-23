import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, Optional

def scrape_reuters_article_data(url: str) -> Optional[Dict[str, str]]:
    """
    指定されたロイターの記事URLから、タイトルと本文を抽出する。

    Args:
        url: ロイターの記事ページのURL。

    Returns:
        dict: 'title'と'content'をキーに持つ辞書。失敗した場合はNoneを返す。
    """
    print(f"Scraping article data from: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        script_tag = soup.find('script', {'id': 'fusion-metadata'})
        if not script_tag or not script_tag.string:
            print("Error: Could not find or read the <script id='fusion-metadata'> tag.")
            return None

        match = re.search(r'Fusion\.globalContent\s*=\s*({.*?});', script_tag.string, re.DOTALL)
        if not match:
            print("Error: Could not find Fusion.globalContent JSON in the script tag.")
            return None

        data = json.loads(match.group(1))
        result = data.get('result', {})

        # タイトルを抽出
        title = result.get('title', '')

        # 本文を抽出
        content_elements = result.get('content_elements', [])
        if not content_elements:
            print("Error: 'content_elements' not found in the JSON data.")
            return None

        article_paragraphs = [
            elem.get('content', '')
            for elem in content_elements
            if elem.get('type') == 'paragraph'
        ]
        content = "\n".join(article_paragraphs)

        if not title or not content:
            print("Warning: Could not extract a complete title and content.")
            return None

        return {
            'title': title.strip(),
            'content': content.strip()
        }

    except requests.exceptions.RequestException as e:
        print(f"Error during requests to {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from script tag: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# # --- テスト実行用のコード ---
# if __name__ == "__main__":
#     target_url = "https://jp.reuters.com/markets/japan/funds/YTRLEPHI2RL6ZKKJVJ575LE7P4-2025-06-20/"
    
#     article_data = scrape_reuters_article_data(target_url)

#     if article_data:
#         print("\n--- Extracted Article Data (from JSON) ---")
#         print(f"Title: {article_data['title']}")
#         print("\n--- Content ---")
#         print(article_data['content'])
#         print("\n---------------------------------------------")
#     else:
#         print("\n--- Failed to extract article data. ---")