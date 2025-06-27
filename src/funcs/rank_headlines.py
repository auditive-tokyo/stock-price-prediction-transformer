import os
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIクライアントを初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rank_headlines_with_openai(
    news_articles: list
) -> list:
    """
    OpenAIのResponses API (Structured Outputs) を使用して、
    ニュース記事リストから影響の大きいヘッドラインをランク付けする。

    Args:
        news_articles: スクレイピングされたニュース記事の辞書のリスト。

    Returns:
        list: 影響が大きいと判断された記事情報の辞書のリスト。
              エラー時は空のリストを返す。
    """
    if not client:
        print("Error: rank_headlines - OpenAI client is not initialized.")
        return []
    if not news_articles:
        print("No news articles provided to rank.")
        return []

    # AIに渡すためにニュースリストを整形
    formatted_news = ""
    for i, article in enumerate(news_articles, 1):
        formatted_news += f"--- Article [{i}] ---\n"
        formatted_news += f"Title: {article.get('title', 'N/A')}\n"
        formatted_news += f"Published: {article.get('published_time', 'N/A')}\n"
        formatted_news += f"Summary: {article.get('summary', 'N/A')}\n"
        formatted_news += f"Link: {article.get('link', 'N/A')}\n\n"

    # --- 変更点: 新しいレスポンス形式のスキーマを定義 ---
    response_format_schema = {
        "type": "object",
        "properties": {
            "ranked_articles": {
                "type": "array",
                "description": "影響の大きい順にランク付けされた、最大5つのニュース記事のリスト。",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer", "description": "影響度の順位 (1が最も影響大)。"},
                        "title": {"type": "string", "description": "ニュース記事のヘッドライン（タイトル）。"},
                        "reason": {"type": "string", "description": "このニュースが市場に大きな影響を与えると判断した簡潔な理由。"},
                        "link": {"type": "string", "description": "元のニュース記事への完全なURL。"}
                    },
                    "required": ["rank", "title", "reason", "link"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["ranked_articles"],
        "additionalProperties": False
    }

    # --- 変更点: AIへの指示をJSON出力に合わせる ---
    system_instructions = (
        "あなたは日本の株式市場を専門とする優秀な金融アナリストです。\n"
        "ユーザーから提供される最新ニュース記事のリストを分析し、短期的な日経225先物の価格に最も大きな影響を与えそうだと考えられるヘッドラインを最大5つまで選び、影響の大きい順にランク付けしてください。\n"
        "評価基準：\n"
        "1. **重要性**: 金融政策の変更、地政学的リスク、主要な経済指標など、市場全体を動かす可能性のあるニュースを最優先してください。\n"
        "2. **鮮度**: できるだけ最新のニュースを重視しますが、数日前のニュースでも市場に織り込まれていない重大な情報であれば選択対象とします。\n"
        "3. **具体性**: 「もみ合い」「一進一退」のような市場の状況を説明するだけのニュースよりも、その原因（例：「中東リスク警戒」「円高が重し」）を明確に示しているニュースを高く評価してください。\n\n"
        "分析が完了したら、必ず提供されたJSONスキーマに従って、結果をJSONオブジェクトとして出力してください。"
    )
    user_input_content = f"--- 最新ニュースリスト ---\n{formatted_news}"

    try:
        # --- 変更点: モデルを gpt-4o に修正し、API呼び出しは変更なし ---
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=system_instructions,
            input=[{"role": "user", "content": user_input_content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "RankedArticleAnalysis",
                    "schema": response_format_schema,
                    "strict": True
                }
            },
            temperature=0.0,
            store=False
        )

        # --- 変更点: vector_search.py を参考にした、より堅牢なレスポンス解析ロジック ---
        if response and hasattr(response, 'output') and isinstance(response.output, list) and len(response.output) > 0:
            # 応答リストの最後にあるはずの 'message' タイプの項目を探す
            output_message_item = next((item for item in reversed(response.output) if hasattr(item, 'type') and item.type == 'message'), None)
            
            if output_message_item and hasattr(output_message_item, 'content') and \
               isinstance(output_message_item.content, list) and len(output_message_item.content) > 0:
                
                # メッセージコンテンツの最初の項目がテキスト出力のはず
                output_text_obj = output_message_item.content[0]
                if hasattr(output_text_obj, 'type') and output_text_obj.type == 'output_text':
                    if hasattr(output_text_obj, 'text') and isinstance(output_text_obj.text, str):
                        try:
                            # モデルが生成したJSON文字列をパース
                            model_generated_data = json.loads(output_text_obj.text)
                            ranked_articles = model_generated_data.get("ranked_articles", [])
                            print("Successfully received and parsed structured JSON response from OpenAI.")
                            return ranked_articles
                        except json.JSONDecodeError as e:
                            print(f"Error: Failed to parse JSON from model output: {e}")
                            print(f"Raw model output: {output_text_obj.text}")
                            return []
                else:
                    print(f"Error: Expected ResponseOutputText, but got: {output_text_obj}")
                    return []
            else:
                print("Error: Could not find valid ResponseOutputMessage content in the response.")
                return []
        else:
            print("Error: Response object or response.output is missing or invalid.")
            return []

    except openai.APIConnectionError as e:
        print(f"OpenAI APIへの接続に失敗しました: {e}")
        return []
    except openai.RateLimitError as e:
        print(f"OpenAI APIのレート制限に達しました: {e}")
        return []
    except openai.APIStatusError as e:
        # エラーレスポンスの詳細を出力
        error_details_str = "N/A"
        try:
            error_details_json = e.response.json()
            error_details_str = json.dumps(error_details_json, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            error_details_str = e.response.text
        print(f"OpenAI APIがエラーステータスを返しました (HTTP {e.status_code}):")
        print(f"エラー詳細:\n{error_details_str}")
        return []
    except Exception as e:
        print(f"OpenAIでのランク付け中に予期せぬエラーが発生しました: {e}")
        return []

# # --- テスト実行コードも新しい出力形式に合わせて更新 ---
# if __name__ == '__main__':
#     print("--- Running test for rank_headlines_with_openai ---")

#     sample_articles = [
#         {"title": "日銀、追加利上げを示唆　円安進行に歯止めか", "published_time": "2025-06-22T10:00:00Z", "summary": "日本銀行の総裁は記者会見で、現在の円安水準と物価上昇を考慮し、次回の金融政策決定会合で追加利上げも選択肢にあると述べた。", "link": "https://jp.reuters.com/markets/japan/funds/SAMPLE-LINK-1"},
#         {"title": "中東情勢が再び緊迫、原油価格が5%急騰", "published_time": "2025-06-22T08:30:00Z", "summary": "ホルムズ海峡での新たな衝突により、地政学的リスクが急激に高まり、WTI原油先物価格は一時1バレル85ドルを突破した。", "link": "https://jp.reuters.com/markets/japan/funds/SAMPLE-LINK-2"},
#         {"title": "米半導体大手、日本に新工場建設を発表　関連銘柄に買い集まる", "published_time": "2025-06-21T15:00:00Z", "summary": "米国の半導体メーカーが、次世代半導体の製造拠点として日本に大規模な工場を建設する計画を発表した。これにより日本の半導体関連株が軒並み上昇した。", "link": "https://jp.reuters.com/markets/japan/funds/SAMPLE-LINK-3"},
#         {"title": "日経平均は小幅反発、方向感に乏しい展開", "published_time": "2025-06-22T06:05:00Z", "summary": "本日の東京株式市場は、海外市場の動向を見極めたいとの雰囲気から、限定的な値動きに終始した。", "link": "https://jp.reuters.com/markets/japan/funds/SAMPLE-LINK-4"}
#     ]

#     ranked_articles_result = rank_headlines_with_openai(sample_articles)

#     if ranked_articles_result:
#         print("\n--- AIによるヘッドラインのランク付け結果 ---")
#         for article in ranked_articles_result:
#             print(f"\n  Rank: {article.get('rank', 'N/A')}")
#             print(f"  Title: {article.get('title', 'N/A')}")
#             print(f"  Reason: {article.get('reason', 'N/A')}")
#             print(f"  Link: {article.get('link', 'N/A')}")
#         print("----------------------------------------------------")
#     else:
#         print("\n--- ヘッドラインのランク付けに失敗しました ---")
