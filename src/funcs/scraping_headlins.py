import requests
from bs4 import BeautifulSoup
import json

def scrape_reuters_nikkei_quote():
    """
    ロイター日経平均ページの<script>タグから埋め込みJSONを抽出し、
    ニュース記事のリストを辞書形式で返す。
    """
    URL = "https://jp.reuters.com/markets/quote/.N225/"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }
    
    extracted_articles = [] # 返却用のリストを初期化

    try:
        response = requests.get(URL, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        script_tag = soup.find('script', id='fusion-metadata')
        if not script_tag:
            print("Error: Could not find the target script tag ('fusion-metadata').")
            return extracted_articles

        script_content = script_tag.string
        json_start_str = "Fusion.globalContent="
        json_start_index = script_content.find(json_start_str)
        
        if json_start_index == -1:
            print("Error: Could not find JSON data within the script tag.")
            return extracted_articles
            
        json_text = script_content[json_start_index + len(json_start_str):].split(';Fusion.globalContentConfig=')[0]
        data = json.loads(json_text)
        articles = data.get('result', {}).get('articles', [])

        if not articles:
            print("No articles found in the JSON data.")
            return extracted_articles

        for article in articles:
            extracted_articles.append({
                "title": article.get('title', 'N/A'),
                "published_time": article.get('published_time', 'N/A'),
                "summary": article.get('description', 'N/A').strip(),
                "link": f"https://jp.reuters.com{article.get('canonical_url', '')}"
            })
        
        print(f"Successfully scraped {len(extracted_articles)} news articles.")
        return extracted_articles

    except Exception as e:
        print(f"\nAn error occurred during scraping: {e}")
        return extracted_articles # エラー時も空のリストを返す
