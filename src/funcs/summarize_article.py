import os
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Optional

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIクライアントを初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_article_with_openai(article_data: Dict[str, str]) -> Optional[str]:
    """
    OpenAIのResponses APIを使用して、記事の全文を市場への影響の観点から要約する。

    Args:
        article_data: 'title'と'content'を含む辞書。

    Returns:
        str: AIによって生成された要約文。エラー時はNoneを返す。
    """
    if not client:
        print("Error: summarize_article - OpenAI client is not initialized.")
        return None
    if not article_data or not article_data.get('title') or not article_data.get('content'):
        print("Error: Invalid article data provided for summarization.")
        return None

    print(f"Summarizing article: '{article_data['title'][:50]}...'")

    title = article_data['title']
    content = article_data['content']

    system_instructions = (
        "あなたは日本の株式市場を専門とする優秀な金融アナリストです。\n"
        "提供されたニュース記事のタイトルと本文を読み、その内容を要約してください。\n"
        "特に、この記事が日経225先物の短期的な価格変動にどのような影響を与えるか（ポジティブ、ネガティブ、ニュートラルなど）という観点を含めて、150字程度で簡潔にまとめてください。\n"
        "出力は、生成した要約文のみにしてください。余計な前置きや結びの言葉は不要です。"
    )
    user_input_content = f"Title: {title}\n\nContent:\n{content}"

    try:
        # Responses APIを呼び出し、平文のテキスト応答を期待する
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=system_instructions,
            input=[{"role": "user", "content": user_input_content}],
            temperature=0.2,
            store=False
        )

        # rank_headlines.py を参考にした堅牢なレスポンス解析ロジック
        if response and hasattr(response, 'output') and isinstance(response.output, list) and len(response.output) > 0:
            output_message_item = next((item for item in reversed(response.output) if hasattr(item, 'type') and item.type == 'message'), None)
            
            if output_message_item and hasattr(output_message_item, 'content') and \
               isinstance(output_message_item.content, list) and len(output_message_item.content) > 0:
                
                output_text_obj = output_message_item.content[0]
                if hasattr(output_text_obj, 'type') and output_text_obj.type == 'output_text':
                    if hasattr(output_text_obj, 'text') and isinstance(output_text_obj.text, str):
                        summary = output_text_obj.text.strip()
                        print("Successfully received summary from OpenAI.")
                        return summary
                else:
                    print(f"Error: Expected ResponseOutputText, but got: {output_text_obj}")
                    return None
            else:
                print("Error: Could not find valid ResponseOutputMessage content in the response.")
                return None
        else:
            print("Error: Response object or response.output is missing or invalid.")
            return None

    except openai.APIConnectionError as e:
        print(f"OpenAI APIへの接続に失敗しました: {e}")
        return None
    except openai.RateLimitError as e:
        print(f"OpenAI APIのレート制限に達しました: {e}")
        return None
    except openai.APIStatusError as e:
        error_details_str = "N/A"
        try:
            error_details_json = e.response.json()
            error_details_str = json.dumps(error_details_json, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            error_details_str = e.response.text
        print(f"OpenAI APIがエラーステータスを返しました (HTTP {e.status_code}):")
        print(f"エラー詳細:\n{error_details_str}")
        return None
    except Exception as e:
        print(f"OpenAIでの要約中に予期せぬエラーが発生しました: {e}")
        return None

# # --- テスト実行用のコード ---
# if __name__ == '__main__':
#     print("--- Running test for summarize_article_with_openai ---")

#     sample_article = {
#         'title': '一進一退、中東リスク警戒　海外マネーや自社株買いが支え＝来週の東京株式市場',
#         'content': (
#             '来週の東京株式市場で日経平均は、一進一退の値動きが想定されている。'
#             'イスラエルとイランの軍事衝突の拡大や米国による介入への警戒感がくすぶっており、株価の上値は抑制されやすい。'
#             '一方、為替がドル高／円安基調にあることや、海外勢の資金の流入、企業による自社株買いへの思惑が投資家心理の支えになり、下値の堅さも意識されそうだ。'
#             '日経平均は３万８５００円を軸にした上下が見込まれる。'
#         )
#     }

#     summary = summarize_article_with_openai(sample_article)

#     if summary:
#         print("\n--- AIによる記事の要約 ---")
#         print(summary)
#         print("--------------------------")
#     else:
#         print("\n--- 記事の要約に失敗しました ---")