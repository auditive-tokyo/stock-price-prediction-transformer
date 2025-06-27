from src.funcs.place_order import place_n225_market_order

def close_n225_position(
    current_position,
    expiration_month,
    host="127.0.0.1",
    port=7497,
    clientId=8
):
    """
    保有ポジションを成り行きで決済する関数。
    current_position: dict（'position'キーに数量、正ならロング・負ならショート）
    expiration_month: 決済する限月（YYYYMM形式）
    """
    qty = abs(current_position.get("position", 0))
    if qty == 0:
        return {"status": "NoPosition", "message": "No position to close."}

    # ロングならSELL、ショートならBUYで決済
    action = "SELL" if current_position["position"] > 0 else "BUY"
    return place_n225_market_order(
        action=action,
        quantity=qty,
        expiration_month=expiration_month,
        host=host,
        port=port,
        clientId=clientId
    )