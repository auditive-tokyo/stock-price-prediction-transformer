import os
import csv
import datetime

def save_analysis_to_csv(analysis_result, situation, csv_file="chart_analysis_result.csv"):
    """
    分析結果をCSVファイルに追記保存する

    Parameters
    ----------
    analysis_result : dict
        分析結果（decision, reason, confidence, additional_info_needed などを含む）
    situation : str
        "決済" または "オーダー" など、状況を示す文字列
    csv_file : str, optional
        保存先CSVファイル名（デフォルト: chart_analysis_result.csv）
    """
    fieldnames = ["datetime", "situation", "decision", "reason", "confidence", "additional_info_needed"]
    now = datetime.datetime.now()
    row = {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "situation": situation,
        "decision": analysis_result.get("decision", ""),
        "reason": analysis_result.get("reason", ""),
        "confidence": analysis_result.get("confidence", ""),
        "additional_info_needed": "; ".join(analysis_result.get("additional_info_needed", []))
    }
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)