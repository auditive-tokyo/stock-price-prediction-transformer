import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIクライアントを初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image_to_base64(image_path):
    """画像をBase64エンコードする"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_close_decision_with_function_calling(
    image_paths, timeframes, current_position, news_summaries, board_info_md, latest_close_price
):
    """
    Function callingを使用して、保有ポジションを「決済する」か「待つ」かを判断させる
    current_position: dict（例: 損益や建玉情報などを要約したもの）
    news_summaries: str または list（AI要約ニュースなど）
    board_info_md: str（板情報のMarkdownテーブル）
    latest_close_price: float または str（最新の終値）
    """
    content = []

    # プロンプト
    prompt = f"""
これらの画像は、日経225先物の{', '.join(timeframes)}のローソク足チャートです。
current_positionには現在保有しているポジションの情報が含まれています。
news_summariesには直近の重要ニュース要約が含まれています。
board_info_mdには最新の板情報がMarkdownテーブル形式で含まれています。
latest_close_priceは直近の終値（最新価格）です。

これらの情報を総合的に分析し、「決済」または「待ち」の判断をしてください。

close_decision関数を使用して判断結果を返してください。

より精度が増すような追加情報が必要な場合は、additional_info_neededフィールドにリストアップしてください。
"""
    content.append({"type": "text", "text": prompt})

    # ポジション情報
    if isinstance(current_position, dict):
        position_info_str = json.dumps(current_position, ensure_ascii=False, indent=2)
    else:
        position_info_str = str(current_position)
    content.append({"type": "text", "text": f"【ポジション情報】\n{position_info_str}"})

    # ニュース要約
    if news_summaries:
        content.append({"type": "text", "text": f"【直近ニュース要約】\n{news_summaries}"})

    # 板情報
    if board_info_md:
        content.append({"type": "text", "text": f"【板情報】\n{board_info_md}"})

    # 最新終値
    if latest_close_price is not None:
        content.append({"type": "text", "text": f"【最新終値】\n{latest_close_price}"})

    # チャート画像
    for path in image_paths:
        base64_image = encode_image_to_base64(path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })

    # ツールの定義
    tools = [
        {
            "type": "function",
            "function": {
                "name": "close_decision",
                "description": "日経225先物のチャートとポジション情報に基づく決済判断",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["決済", "待ち"],
                            "description": "チャートとポジション情報に基づく決済判断"
                        },
                        "reason": {
                            "type": "string",
                            "description": "判断の理由と詳細な分析結果"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["low", "mid", "high"],
                            "description": "判断の確信度"
                        },
                        "additional_info_needed": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "より精度の高い判断のために必要な追加情報のリスト"
                        }
                    },
                    "required": ["decision", "reason", "confidence"]
                }
            }
        }
    ]

    # APIリクエスト
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "close_decision"}}
    )

    # 結果を取得
    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.type == 'function' and tool_call.function.name == 'close_decision':
        function_args = json.loads(tool_call.function.arguments)
        return function_args
    else:
        return {"error": "Function call failed or not available"}