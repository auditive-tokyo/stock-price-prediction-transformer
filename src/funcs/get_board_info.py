import os
import json

def get_board_info_markdown():
    # プロジェクトルートのboard_info/board_info.jsonを参照
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    board_info_path = os.path.join(project_root, "board_info", "board_info.json")
    with open(board_info_path, "r", encoding="utf-8") as f:
        board = json.load(f)

    md_table = "| 売気配 | 値段 | 買気配 |\n|-------|------|-------|\n"
    for item in board:
        sell = item["sell_qty"] if item["sell_qty"] is not None else ""
        price = item["price"] if item["price"] is not None else ""
        buy = item["buy_qty"] if item["buy_qty"] is not None else ""
        md_table += f"| {sell} | {price} | {buy} |\n"

    return md_table