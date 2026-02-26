"""Discord embed constructors for each notification event type.

All visual formatting logic lives here — discord_client.py never
formats anything itself. Each builder returns a dict matching the
Discord embed JSON structure.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


PREDICTION_DISCLAIMER = (
    "This is a statistical estimate based on historical pattern similarity, "
    "not a guaranteed prediction. Past signal patterns do not guarantee "
    "future price movements."
)

# Layer name to embed color mapping
LAYER_COLORS = {
    "L1_VWAP_MR": 0x3498DB,      # blue
    "L2_ORB": 0x2ECC71,           # green
    "L3_RSI_SCALP": 0xF39C12,     # orange
    "L4_VOL_SURGE": 0x9B59B6,     # purple
    "EXIT": 0xE74C3C,              # red
    "SYSTEM": 0x95A5A6,            # grey
    "CIRCUIT_BREAKER": 0xFF0000,   # bright red
    "PDT_BLOCK": 0xFF6600,         # orange-red
    "DAILY": 0x1ABC9C,             # teal
}

LAYER_TITLES = {
    "L1_VWAP_MR": "VWAP MR ENTRY",
    "L2_ORB": "ORB BREAKOUT ENTRY",
    "L3_RSI_SCALP": "RSI SCALP ENTRY",
    "L4_VOL_SURGE": "VOL SURGE ENTRY",
}

LAYER_EMOJIS = {
    "L1_VWAP_MR": "\U0001f4ca",    # 📊
    "L2_ORB": "\U0001f680",         # 🚀
    "L3_RSI_SCALP": "\u26a1",       # ⚡
    "L4_VOL_SURGE": "\U0001f30a",   # 🌊
}

ALERT_COLORS = {
    "CIRCUIT_BREAKER_TRIGGERED": 0xFF0000,
    "PDT_BLOCK": 0xFF6600,
    "STOP_LOSS_HIT": 0xFF4444,
    "TRAILING_STOP_HIT": 0xFF8800,
    "CIRCUIT_BREAKER_RESET": 0x00FF88,
    "DB_FALLBACK_SQLITE": 0xFFFF00,
    "SYSTEM_RESTART": 0x0088FF,
    "REGIME_CHANGE": 0xAA00FF,
    "MARKET_CLOSED": 0x888888,
    "ERROR": 0xFF0000,
    "WS_RECONNECT": 0x0088FF,
}


def _timestamp_now() -> str:
    """Return ISO 8601 UTC timestamp for embed footer.

    Returns:
        str: Current UTC time in ISO 8601 format.

    Example:
        >>> _timestamp_now()
        '2024-01-15T10:32:07.000Z'
    """
    return datetime.now(timezone.utc).isoformat()


def _footer_text(mode: str = "Paper") -> str:
    """Build the standard embed footer text.

    Args:
        mode: Trading mode label. Default 'Paper'.

    Returns:
        str: Footer text string.

    Example:
        >>> _footer_text()
        'AlgoTrader Paper'
    """
    return f"AlgoTrader {mode}"


def _wrap_embed(embed_obj: dict) -> dict:
    """Wrap a single embed object in the Discord webhook payload structure.

    Args:
        embed_obj: Single Discord embed dict.

    Returns:
        dict: {'embeds': [embed_obj]} ready for the Discord webhook API.
    """
    return {"embeds": [embed_obj]}


def _format_hold_minutes(minutes: float) -> str:
    """Format hold duration as a concise minutes string.

    Args:
        minutes: Hold time in minutes.

    Returns:
        str: E.g. '47 minutes' or '1 minute'.
    """
    minutes = int(round(minutes))
    if minutes == 1:
        return "1 minute"
    return f"{minutes} minutes"


def build_trade_entry_embed(
    trade_data: dict,
    prediction_result=None,
) -> dict:
    """Build a Discord embed for a trade entry event.

    Selects layer-specific title and color based on strategy/layer name.

    Args:
        trade_data: Dict with keys: symbol, entry_price, shares, stop_price,
            strategy_name, and optional z_score, rsi, volume_ratio, atr,
            total_equity, regime_scalar, confidence.
        prediction_result: Optional PredictionResult or dict.

    Returns:
        dict: Bare Discord embed dict (caller wraps in {"embeds": [...]}).
    """
    symbol = trade_data.get("symbol", "???")
    layer_name = trade_data.get("strategy_name", "")
    signal_price = float(trade_data.get("entry_price", 0.0) or 0.0)
    stop_price = float(trade_data.get("stop_price", 0.0) or 0.0)
    confidence = float(trade_data.get("confidence", 0.0) or 0.0)
    shares = int(trade_data.get("shares", 0) or 0)
    metadata = trade_data

    color = LAYER_COLORS.get(layer_name, 0x3498DB)
    emoji = LAYER_EMOJIS.get(layer_name, "\U0001f4c8")
    layer_title = LAYER_TITLES.get(layer_name, f"{layer_name} ENTRY")

    position_size = signal_price * shares
    stop_risk_pct = ((stop_price - signal_price) / signal_price * 100) if signal_price > 0 and stop_price > 0 else 0.0
    stop_risk_dollar = (stop_price - signal_price) * shares if stop_price > 0 else 0.0

    fields = [
        {"name": "Symbol", "value": symbol, "inline": True},
        {"name": "Layer", "value": layer_name, "inline": True},
        {"name": "Order Type", "value": "Market", "inline": True},
        {"name": "Entry Price", "value": f"${signal_price:.2f}", "inline": True},
        {"name": "Shares", "value": str(shares), "inline": True},
        {"name": "Position Size", "value": f"${position_size:.2f}", "inline": True},
    ]

    if stop_price > 0:
        fields.append({
            "name": "Stop Loss",
            "value": f"${stop_price:.2f} ({stop_risk_pct:.1f}% / ${stop_risk_dollar:.2f})",
            "inline": True,
        })

    fields.append({"name": "Confidence", "value": f"{confidence * 100:.1f}%", "inline": True})

    # Layer-specific metadata fields
    vwap = metadata.get("vwap", 0)
    rsi = metadata.get("rsi", 0)
    z_score = metadata.get("z_score", 0)
    volume_ratio = metadata.get("volume_ratio", 0)
    atr = metadata.get("atr", 0)

    if vwap:
        fields.append({"name": "VWAP", "value": f"${vwap:.2f}", "inline": True})
    if rsi:
        fields.append({"name": "RSI", "value": f"{rsi:.1f}", "inline": True})
    if z_score:
        fields.append({"name": "Z-Score", "value": f"{z_score:.3f}", "inline": True})
    if volume_ratio:
        fields.append({"name": "Volume Ratio", "value": f"{volume_ratio:.2f}x avg", "inline": True})
    if atr:
        fields.append({"name": "ATR", "value": f"${atr:.4f}", "inline": True})

    # Prediction section
    if prediction_result is not None:
        if hasattr(prediction_result, "direction"):
            direction = prediction_result.direction
            pred_confidence = prediction_result.confidence_pct
            est_low = prediction_result.estimated_low
            est_high = prediction_result.estimated_high
            key_driver = prediction_result.key_driver
            bars_horizon = getattr(prediction_result, "bars_horizon", 5)
        elif isinstance(prediction_result, dict):
            direction = prediction_result.get("direction", "NEUTRAL")
            pred_confidence = prediction_result.get("confidence_pct", 0)
            est_low = prediction_result.get("estimated_low", 0)
            est_high = prediction_result.get("estimated_high", 0)
            key_driver = prediction_result.get("key_driver", "N/A")
            bars_horizon = prediction_result.get("bars_horizon", 5)
        else:
            direction = pred_confidence = est_low = est_high = key_driver = bars_horizon = None

        if direction is not None:
            fields.extend([
                {"name": f"\U0001f52e Forecast (Next {bars_horizon} Bars)", "value": "\u200b", "inline": False},
                {"name": "Direction", "value": str(direction), "inline": True},
                {"name": "Est. Range", "value": f"${est_low:.2f} – ${est_high:.2f}", "inline": True},
                {"name": "Model Confidence", "value": f"{pred_confidence:.1f}%", "inline": True},
                {"name": "Key Driver", "value": str(key_driver), "inline": False},
            ])

    fields.append({"name": "\u26a0\ufe0f Disclaimer", "value": PREDICTION_DISCLAIMER, "inline": False})

    embed = {
        "title": f"{emoji} {layer_title} \u2014 {symbol}",
        "color": color,
        "fields": fields,
        "footer": {"text": _footer_text()},
        "timestamp": _timestamp_now(),
    }
    return embed


def build_trade_exit_embed(trade: dict) -> dict:
    """Build a Discord embed for a trade exit event.

    Hold duration is displayed in MINUTES (e.g. "47 minutes").

    Args:
        trade: Dict with symbol, layer_name, entry_price, exit_price,
               qty, pnl, hold_minutes, exit_reason, slippage.

    Returns:
        dict: Discord webhook payload {'embeds': [...]}.
    """
    symbol = trade.get("symbol", "???")
    layer_name = trade.get("layer_name", "")
    entry_price = trade.get("entry_price", 0.0)
    exit_price = trade.get("exit_price", 0.0)
    qty = trade.get("qty", 0)
    pnl = trade.get("pnl", 0.0)
    hold_minutes = trade.get("hold_minutes", 0.0)
    exit_reason = trade.get("exit_reason", "SIGNAL")
    slippage = trade.get("slippage", 0.0)
    entry_time = trade.get("entry_time", "")
    exit_time = trade.get("exit_time", "")

    if pnl > 0:
        title_emoji = "\U0001f7e2"
        title_label = "PROFITABLE EXIT"
        color = 0x00FF00
        pnl_emoji = "\u2705"
    elif pnl < 0:
        title_emoji = "\U0001f534"
        title_label = "LOSS EXIT"
        color = 0xFF0000
        pnl_emoji = "\u274c"
    else:
        title_emoji = "\u26aa"
        title_label = "BREAKEVEN EXIT"
        color = 0x888888
        pnl_emoji = "\u2796"

    hold_str = _format_hold_minutes(hold_minutes)

    # Format timestamps
    if isinstance(entry_time, datetime):
        entry_str = entry_time.strftime("%Y-%m-%d %H:%M ET")
    else:
        entry_str = str(entry_time)
    if isinstance(exit_time, datetime):
        exit_str = exit_time.strftime("%Y-%m-%d %H:%M ET")
    else:
        exit_str = str(exit_time)

    pnl_per_share = (exit_price - entry_price)

    fields = [
        {"name": "Symbol", "value": symbol, "inline": True},
        {"name": "Layer", "value": layer_name or "N/A", "inline": True},
        {"name": "Exit Reason", "value": exit_reason, "inline": True},
        {"name": "Entry Price", "value": f"${entry_price:.2f}", "inline": True},
        {"name": "Exit Price", "value": f"${exit_price:.2f}", "inline": True},
        {"name": "P&L Per Share", "value": f"${pnl_per_share:+.4f}", "inline": True},
        {"name": "Shares", "value": str(qty), "inline": True},
        {"name": "Net P&L", "value": f"${pnl:+.2f} {pnl_emoji}", "inline": True},
        {"name": "Slippage Cost", "value": f"${slippage:.4f}", "inline": True},
        {"name": "Hold Duration", "value": hold_str, "inline": True},
        {"name": "Entry Time", "value": entry_str, "inline": True},
        {"name": "Exit Time", "value": exit_str, "inline": True},
    ]

    embed = {
        "title": f"{title_emoji} {title_label} \u2014 {symbol}",
        "color": color,
        "fields": fields,
        "footer": {"text": _footer_text()},
        "timestamp": _timestamp_now(),
    }
    return embed


def build_signal_embed(
    signal,
    prediction_result=None,
) -> dict:
    """Build a Discord embed for a detected signal (pre-order notification).

    Args:
        signal: SignalResult object with symbol, signal, layer_name,
                signal_price, confidence, metadata.
        prediction_result: Optional PredictionResult or dict.

    Returns:
        dict: Discord webhook payload {'embeds': [...]}.
    """
    symbol = getattr(signal, "symbol", "???")
    signal_type = getattr(signal, "signal", "HOLD")
    layer_name = getattr(signal, "layer_name", "")
    current_price = getattr(signal, "signal_price", 0.0)
    confidence = getattr(signal, "confidence", 0.0)
    metadata = getattr(signal, "metadata", {}) or {}

    regime = metadata.get("regime", "UNKNOWN")
    z_score = metadata.get("z_score", 0.0)
    rsi_val = metadata.get("rsi", 0.0)
    volume_ratio = metadata.get("volume_ratio", 1.0)
    atr = metadata.get("atr", 0.0)

    regime_emoji = {"BULL": "\U0001f7e2", "NEUTRAL": "\U0001f7e1", "BEAR": "\U0001f534"}.get(regime, "\u2753")

    fields = [
        {"name": "Symbol", "value": symbol, "inline": True},
        {"name": "Signal", "value": f"{signal_type} ({layer_name})", "inline": True},
        {"name": "Confidence", "value": f"{confidence * 100:.1f}%", "inline": True},
        {"name": "Current Price", "value": f"${current_price:.2f}", "inline": True},
        {"name": "Regime", "value": f"{regime_emoji} {regime}", "inline": True},
    ]

    if z_score:
        fields.append({"name": "Z-Score", "value": f"{z_score:.3f}", "inline": True})
    if rsi_val:
        fields.append({"name": "RSI", "value": f"{rsi_val:.1f}", "inline": True})
    if volume_ratio and volume_ratio != 1.0:
        fields.append({"name": "Volume vs Avg", "value": f"{volume_ratio:.2f}x", "inline": True})
    if atr:
        fields.append({"name": "ATR", "value": f"${atr:.4f}", "inline": True})

    if prediction_result is not None:
        if hasattr(prediction_result, "direction"):
            direction = prediction_result.direction
            pred_confidence = prediction_result.confidence_pct
            est_low = prediction_result.estimated_low
            est_high = prediction_result.estimated_high
            key_driver = prediction_result.key_driver
            bars_horizon = getattr(prediction_result, "bars_horizon", 5)
        elif isinstance(prediction_result, dict):
            direction = prediction_result.get("direction", "NEUTRAL")
            pred_confidence = prediction_result.get("confidence_pct", 0)
            est_low = prediction_result.get("estimated_low", 0)
            est_high = prediction_result.get("estimated_high", 0)
            key_driver = prediction_result.get("key_driver", "N/A")
            bars_horizon = prediction_result.get("bars_horizon", 5)
        else:
            direction = None

        if direction is not None:
            fields.extend([
                {"name": f"\U0001f52e Model Forecast (Next {bars_horizon} Bars)", "value": "\u200b", "inline": False},
                {"name": "Direction", "value": str(direction), "inline": True},
                {"name": "Est. Range", "value": f"${est_low:.2f} – ${est_high:.2f}", "inline": True},
                {"name": "Confidence", "value": f"{pred_confidence:.1f}%", "inline": True},
                {"name": "Key Driver", "value": str(key_driver), "inline": False},
            ])

    fields.append({"name": "\u26a0\ufe0f Disclaimer", "value": PREDICTION_DISCLAIMER, "inline": False})

    color = LAYER_COLORS.get(layer_name, 0x7B2FBE)

    embed = {
        "title": f"\U0001f4e1 SIGNAL DETECTED \u2014 {symbol}",
        "color": color,
        "fields": fields,
        "footer": {"text": _footer_text()},
        "timestamp": _timestamp_now(),
    }
    return embed


def build_pdt_block_embed(
    symbol: str,
    entry_time,
    unrealized_pnl: float,
    pdt_count: int,
    reset_date,
) -> dict:
    """Build a Discord embed for a PDT-blocked exit notification.

    Args:
        symbol: Ticker symbol of the blocked position.
        entry_time: Entry datetime (ET or UTC).
        unrealized_pnl: Current unrealized P&L in dollars.
        pdt_count: Current rolling day trade count (e.g. 3).
        reset_date: Date when the oldest day trade expires (rolling window resets).

    Returns:
        dict: Discord webhook payload {'embeds': [...]}.
    """
    max_day_trades = 3

    if isinstance(entry_time, datetime):
        entry_str = entry_time.strftime("%Y-%m-%d %H:%M ET")
    else:
        entry_str = str(entry_time)

    if hasattr(reset_date, "strftime"):
        reset_str = reset_date.strftime("%Y-%m-%d")
    else:
        reset_str = str(reset_date)

    pnl_emoji = "\u2705" if unrealized_pnl >= 0 else "\u274c"

    fields = [
        {"name": "Symbol", "value": symbol, "inline": True},
        {
            "name": "Reason",
            "value": f"Rolling day trade limit reached {pdt_count}/{max_day_trades}",
            "inline": True,
        },
        {"name": "Entry Time (ET)", "value": entry_str, "inline": True},
        {
            "name": "Current P&L (Unrealized)",
            "value": f"${unrealized_pnl:+.2f} {pnl_emoji}",
            "inline": True,
        },
        {
            "name": "Action",
            "value": "Position held overnight \u2014 exits tomorrow",
            "inline": True,
        },
        {
            "name": "PDT Count",
            "value": f"{pdt_count}/{max_day_trades} day trades used",
            "inline": True,
        },
        {"name": "Resets On", "value": reset_str, "inline": True},
    ]

    embed = {
        "title": f"\u23f8\ufe0f PDT BLOCK \u2014 {symbol} EXIT DEFERRED",
        "color": LAYER_COLORS["PDT_BLOCK"],
        "fields": fields,
        "footer": {"text": _footer_text()},
        "timestamp": _timestamp_now(),
    }
    return embed


def build_circuit_breaker_embed(
    reason: str,
    portfolio_value: float,
    trigger_type: str,
) -> dict:
    """Build a Discord embed for a circuit breaker activation.

    Args:
        reason: Human-readable description of why the breaker fired.
        portfolio_value: Current portfolio value at the time of trigger.
        trigger_type: Type of trigger (e.g. 'DAILY_LOSS', 'DRAWDOWN', 'MANUAL').

    Returns:
        dict: Discord webhook payload {'embeds': [...]}.
    """
    fields = [
        {"name": "Trigger Type", "value": trigger_type, "inline": True},
        {"name": "Portfolio Value", "value": f"${portfolio_value:.2f}", "inline": True},
        {"name": "Reason", "value": reason, "inline": False},
        {
            "name": "Effect",
            "value": "All trading halted. Open positions will be liquidated. Manual intervention required to resume.",
            "inline": False,
        },
        {"name": "Timestamp", "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "inline": True},
    ]

    embed = {
        "title": "\U0001f6a8 CIRCUIT BREAKER TRIGGERED \u2014 ALL TRADING HALTED",
        "color": LAYER_COLORS["CIRCUIT_BREAKER"],
        "fields": fields,
        "footer": {"text": _footer_text()},
        "timestamp": _timestamp_now(),
    }
    return embed


def build_daily_summary_embed(summary_dict: dict) -> dict:
    """Build a Discord embed for the end-of-day portfolio summary.

    Args:
        summary_dict: Dict with portfolio_value, starting_capital, daily_pnl,
            daily_pnl_pct, total_trades, winning_trades, losing_trades,
            win_rate, eod_closes, pdt_blocks, regime, sharpe, sortino,
            max_drawdown, peak_value, layer_breakdown (dict of layer->stats).

    Returns:
        dict: Discord webhook payload {'embeds': [...]}.
    """
    report_date = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    starting = summary_dict.get("starting_capital", 100000.0)
    ending = summary_dict.get("portfolio_value", starting)
    daily_pnl = summary_dict.get("daily_pnl", ending - starting)
    daily_pct = (daily_pnl / starting * 100) if starting > 0 else 0.0
    all_time_pnl = ending - starting
    all_time_pct = (all_time_pnl / starting * 100) if starting > 0 else 0.0

    total_trades = summary_dict.get("total_trades", 0)
    winning_trades = summary_dict.get("winning_trades", 0)
    losing_trades = summary_dict.get("losing_trades", 0)
    win_rate = summary_dict.get("win_rate", 0.0)
    eod_closes = summary_dict.get("eod_closes", 0)
    pdt_blocks = summary_dict.get("pdt_blocks", 0)

    regime = summary_dict.get("regime", "UNKNOWN")
    regime_emoji = {"BULL": "\U0001f7e2", "NEUTRAL": "\U0001f7e1", "BEAR": "\U0001f534"}.get(regime, "\u2753")

    sharpe = summary_dict.get("sharpe", 0.0)
    sortino = summary_dict.get("sortino", 0.0)
    max_dd = summary_dict.get("max_drawdown", 0.0)
    peak_value = summary_dict.get("peak_value", ending)

    layer_breakdown = summary_dict.get("layer_breakdown", {})

    fields = [
        {"name": "\U0001f4bc Portfolio", "value": "\u200b", "inline": False},
        {"name": "Starting Balance", "value": f"${starting:.2f}", "inline": True},
        {"name": "Ending Balance", "value": f"${ending:.2f}", "inline": True},
        {"name": "Today's P&L", "value": f"${daily_pnl:+.2f} ({daily_pct:+.2f}%)", "inline": True},
        {"name": "All-Time P&L", "value": f"${all_time_pnl:+.2f} ({all_time_pct:+.2f}%)", "inline": True},
        {"name": "Peak Equity", "value": f"${peak_value:.2f}", "inline": True},
        {"name": "\U0001f4ca Activity", "value": "\u200b", "inline": False},
        {"name": "Trades Closed", "value": str(total_trades), "inline": True},
        {"name": "Wins / Losses", "value": f"{winning_trades} / {losing_trades}", "inline": True},
        {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
        {"name": "EOD Forced Closes", "value": str(eod_closes), "inline": True},
        {"name": "PDT Blocks", "value": str(pdt_blocks), "inline": True},
        {"name": "\U0001f4c8 Risk Metrics", "value": "\u200b", "inline": False},
        {"name": "Sharpe Ratio", "value": f"{sharpe:.4f}", "inline": True},
        {"name": "Sortino Ratio", "value": f"{sortino:.4f}", "inline": True},
        {"name": "Max Drawdown", "value": f"{max_dd:.2f}%", "inline": True},
        {"name": "\U0001f321\ufe0f Market Conditions", "value": "\u200b", "inline": False},
        {"name": "Regime", "value": f"{regime_emoji} {regime}", "inline": True},
    ]

    if layer_breakdown:
        layer_lines = []
        for layer_name, stats in layer_breakdown.items():
            t = stats.get("trades", 0)
            wr = stats.get("win_rate", 0.0)
            pnl = stats.get("total_pnl", 0.0)
            layer_lines.append(f"{layer_name}: {t} trades, {wr:.0f}% WR, ${pnl:+.2f}")
        if layer_lines:
            fields.append({
                "name": "\U0001f4cb Layer Breakdown",
                "value": "\n".join(layer_lines),
                "inline": False,
            })

    embed = {
        "title": f"\U0001f4c5 DAILY SUMMARY \u2014 {report_date}",
        "color": LAYER_COLORS["DAILY"],
        "fields": fields,
        "footer": {"text": _footer_text()},
        "timestamp": _timestamp_now(),
    }
    return embed


def build_system_embed(
    event_type: str,
    message: str,
    extra_fields: Optional[List[dict]] = None,
) -> dict:
    """Build a Discord embed for a system event (startup, restart, reconnect, error).

    Args:
        event_type: One of 'STARTUP', 'RESTART', 'WS_RECONNECT', 'ERROR',
                    'CIRCUIT_BREAKER_RESET', 'REGIME_CHANGE', 'MARKET_CLOSED', etc.
        message: Human-readable description of the event.
        extra_fields: Optional list of additional field dicts to append.

    Returns:
        dict: Discord webhook payload {'embeds': [...]}.
    """
    title_map = {
        "STARTUP": "\U0001f7e2 SYSTEM STARTED",
        "RESTART": "\U0001f504 SYSTEM RESTART",
        "WS_RECONNECT": "\U0001f504 WEBSOCKET RECONNECTED",
        "WS_DISCONNECT": "\u26a0\ufe0f WEBSOCKET DISCONNECTED",
        "ERROR": "\u274c SYSTEM ERROR",
        "CIRCUIT_BREAKER_RESET": "\u2705 CIRCUIT BREAKER RESET",
        "REGIME_CHANGE": "\U0001f30a MARKET REGIME CHANGE",
        "MARKET_CLOSED": "\U0001f319 MARKET CLOSED",
        "DB_FALLBACK_SQLITE": "\u26a0\ufe0f DATABASE FALLBACK TO SQLITE",
        "INFO": "\u2139\ufe0f SYSTEM INFO",
    }
    color_map = {
        "STARTUP": 0x2ECC71,
        "RESTART": ALERT_COLORS["SYSTEM_RESTART"],
        "WS_RECONNECT": ALERT_COLORS["WS_RECONNECT"],
        "WS_DISCONNECT": 0xFF8800,
        "ERROR": 0xFF0000,
        "CIRCUIT_BREAKER_RESET": 0x00FF88,
        "REGIME_CHANGE": 0xAA00FF,
        "MARKET_CLOSED": 0x888888,
        "DB_FALLBACK_SQLITE": 0xFFFF00,
        "INFO": LAYER_COLORS["SYSTEM"],
    }

    title = title_map.get(event_type, f"\u26a0\ufe0f SYSTEM: {event_type}")
    color = color_map.get(event_type, LAYER_COLORS["SYSTEM"])

    fields = [
        {"name": "Event", "value": event_type, "inline": True},
        {
            "name": "Time (UTC)",
            "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "inline": True,
        },
        {"name": "Details", "value": message, "inline": False},
    ]

    if extra_fields:
        fields.extend(extra_fields)

    embed = {
        "title": title,
        "color": color,
        "fields": fields,
        "footer": {"text": _footer_text()},
        "timestamp": _timestamp_now(),
    }
    return embed


def build_alert_embed(alert_type: str, message: str, data: Optional[dict] = None) -> dict:
    """Build a Discord embed for a generic alert notification.

    Preserved for backward compatibility with existing code that calls build_alert_embed.

    Args:
        alert_type: Alert type key from ALERT_COLORS.
        message: Human-readable alert message.
        data: Optional dictionary with additional alert context fields.

    Returns:
        dict: Discord embed payload ready for sending (bare embed, not wrapped).
    """
    color = ALERT_COLORS.get(alert_type, 0xFF0000)
    data = data or {}

    title_map = {
        "CIRCUIT_BREAKER_TRIGGERED": "\U0001f6a8 CIRCUIT BREAKER TRIGGERED \u2014 ALL TRADING HALTED",
        "PDT_BLOCK": "\u23f8\ufe0f PDT BLOCK \u2014 EXIT DENIED",
        "STOP_LOSS_HIT": "\U0001f534 STOP LOSS HIT",
        "TRAILING_STOP_HIT": "\U0001f7e0 TRAILING STOP HIT",
        "CIRCUIT_BREAKER_RESET": "\u2705 CIRCUIT BREAKER RESET",
        "DB_FALLBACK_SQLITE": "\u26a0\ufe0f DATABASE FALLBACK TO SQLITE",
        "SYSTEM_RESTART": "\U0001f504 SYSTEM RESTART",
        "REGIME_CHANGE": "\U0001f30a MARKET REGIME CHANGE",
        "MARKET_CLOSED": "\U0001f319 MARKET CLOSED",
        "ERROR": "\u274c SYSTEM ERROR",
        "WS_RECONNECT": "\U0001f504 WEBSOCKET RECONNECTED",
    }
    title = title_map.get(alert_type, f"\u26a0\ufe0f ALERT: {alert_type}")

    fields = [{"name": "Details", "value": message, "inline": False}]

    for key, value in data.items():
        if key not in ("alert_type",):
            fields.append({
                "name": key.replace("_", " ").title(),
                "value": str(value),
                "inline": True,
            })

    return {
        "title": title,
        "color": color,
        "fields": fields,
        "footer": {"text": _footer_text()},
        "timestamp": _timestamp_now(),
    }
