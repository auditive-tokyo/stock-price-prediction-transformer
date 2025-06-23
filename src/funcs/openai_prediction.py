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

def analyze_chart_with_function_calling(image_paths, timeframes):
    """Function callingを使用して複数のチャート分析を構造化する"""
    # メッセージに含めるコンテンツリストを作成
    content = []
    
    # TODO: 将来的に板情報を取得してプロンプトに追加する
    # board_info = "..."

    # プロンプト
    prompt = f"""
これらの画像は、日経225先物の{', '.join(timeframes)}のローソク足チャートです。
これらの情報を総合的に分析し、「買い、売り、待ち」の判断をしてください。

trading_decision関数を使用して判断結果を返してください。

より精度が増すような追加情報が必要な場合は、additional_info_neededフィールドにリストアップしてください。
"""
    content.append({"type": "text", "text": prompt})

    # 各画像をBase64エンコードしてコンテンツに追加
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
                "name": "trading_decision",
                "description": "日経225先物のチャート分析に基づく取引判断",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["買い", "売り", "待ち"],
                            "description": "チャート情報に基づく取引判断"
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
        tool_choice={"type": "function", "function": {"name": "trading_decision"}}
    )
    
    # 結果を取得
    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.type == 'function' and tool_call.function.name == 'trading_decision':
        function_args = json.loads(tool_call.function.arguments)
        return function_args
    else:
        return {"error": "Function call failed or not available"}
