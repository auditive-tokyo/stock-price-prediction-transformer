import time
from typing import List, Dict, Optional
from .scraping_headlins import scrape_reuters_nikkei_quote
from .rank_headlines import rank_headlines_with_openai
from .scrape_reuters_article_data import scrape_reuters_article_data
from .summarize_article import summarize_article_with_openai

def get_and_summarize_top_articles(top_n: int = 5) -> Optional[List[str]]:
    """
    ロイターからヘッドラインを取得・ランク付けし、上位記事をスクレイピングして要約する。

    Args:
        top_n: 処理対象とする上位記事の数。

    Returns:
        各記事の要約文（文字列）のリスト。失敗した場合はNoneを返す。
    """
    print("\n--- Step: Fetching, Ranking, Scraping, and Summarizing News Articles ---")
    
    # 1. ヘッドラインとURLを取得
    print("Fetching news headlines...")
    news_headlines = scrape_reuters_nikkei_quote()
    if not news_headlines:
        print("Could not fetch news headlines.")
        return None

    # 2. OpenAIでヘッドラインをランク付け
    print("Ranking headlines with OpenAI...")
    ranked_headlines = rank_headlines_with_openai(news_headlines)
    print(f"Ranked Headlines: {ranked_headlines}")
    if not ranked_headlines:
        print("Could not rank headlines.")
        return None
    
    print(f"Top {top_n} ranked headlines identified.")

    # 3. 上位N件の記事をスクレイピング & 要約
    summaries_list = [] # ▼▼▼【修正点】要約文を直接格納するリスト
    print(f"\nProcessing top {top_n} articles...")
    for i, headline in enumerate(ranked_headlines[:top_n], 1):
        url = headline.get('link')
        if not url:
            continue
        
        print(f"\n--- Processing Article {i}/{top_n} ---")
        article_data = scrape_reuters_article_data(url)
        
        if article_data:
            summary = summarize_article_with_openai(article_data)
            
            if summary:
                # ▼▼▼【修正点】要約文を直接リストに追加
                summaries_list.append(summary)
                print(f"  -> Summary received and added to list.")
            else:
                print(f"  -> Failed to summarize article: '{article_data['title'][:30]}...'")
        else:
            print(f"  -> Failed to scrape content from: {url}")

        if i < top_n:
            print("Waiting for 1 second...")
            time.sleep(2)
        
    if not summaries_list:
        print("Failed to generate any summaries.")
        return None

    # ▼▼▼【修正点】要約文のリストを返す
    return summaries_list