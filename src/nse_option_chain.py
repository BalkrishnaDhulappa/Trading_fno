"""Fetch NIFTY option chain snapshots from NSE India."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

NSE_BASE = "https://www.nseindia.com"
OC_PAGE = f"{NSE_BASE}/option-chain"
CONTRACT_INFO = f"{NSE_BASE}/api/option-chain-contract-info"
OC_V3 = f"{NSE_BASE}/api/option-chain-v3"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    # Warm cookies via the option-chain page (homepage often 403).
    session.get(OC_PAGE, timeout=30)
    time.sleep(0.4)
    return session


def fetch_expiries(symbol: str = "NIFTY") -> list[str]:
    session = _session()
    resp = session.get(
        CONTRACT_INFO,
        params={"symbol": symbol},
        headers={**DEFAULT_HEADERS, "Accept": "application/json", "Referer": OC_PAGE},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return list(data.get("expiryDates") or [])


def fetch_option_chain(
    symbol: str = "NIFTY",
    expiry: str | None = None,
    chain_type: str = "Indices",
) -> dict[str, Any]:
    """Return raw NSE option-chain-v3 JSON for one expiry."""
    session = _session()
    headers = {**DEFAULT_HEADERS, "Accept": "application/json", "Referer": OC_PAGE}
    if expiry is None:
        expiries = fetch_expiries(symbol)
        if not expiries:
            raise RuntimeError(f"No expiries returned for {symbol}")
        expiry = expiries[0]
        # Reuse a fresh session after contract-info call.
        session = _session()

    resp = session.get(
        OC_V3,
        params={"type": chain_type, "symbol": symbol, "expiry": expiry},
        headers=headers,
        timeout=45,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload or not payload.get("records"):
        raise RuntimeError("Empty option chain response from NSE")
    payload["_meta"] = {
        "symbol": symbol,
        "expiry": expiry,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def option_chain_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    records = payload.get("records") or {}
    rows: list[dict[str, Any]] = []
    for item in records.get("data") or []:
        strike = item.get("strikePrice")
        for side in ("CE", "PE"):
            leg = item.get(side)
            if not leg:
                continue
            rows.append(
                {
                    "strike": strike,
                    "side": side,
                    "expiry": leg.get("expiryDate") or item.get("expiryDates"),
                    "ltp": leg.get("lastPrice"),
                    "change": leg.get("change"),
                    "pct_change": leg.get("pChange"),
                    "oi": leg.get("openInterest"),
                    "chg_oi": leg.get("changeinOpenInterest"),
                    "volume": leg.get("totalTradedVolume"),
                    "iv": leg.get("impliedVolatility"),
                    "spot": records.get("underlyingValue"),
                    "timestamp": records.get("timestamp"),
                }
            )
    frame = pd.DataFrame(rows)
    meta = payload.get("_meta") or {}
    if not frame.empty:
        frame["symbol"] = meta.get("symbol", "NIFTY")
        frame["selected_expiry"] = meta.get("expiry")
        frame["fetched_at"] = meta.get("fetched_at")
    return frame


def summarize_option_chain(payload: dict[str, Any]) -> dict[str, float]:
    """Aggregate OC features useful for next-day strategy models."""
    records = payload.get("records") or {}
    spot = float(records.get("underlyingValue") or 0.0)
    data = records.get("data") or []
    if not data or spot <= 0:
        raise RuntimeError("Cannot summarize empty option chain")

    strikes = sorted(float(r["strikePrice"]) for r in data if r.get("strikePrice") is not None)
    atm = min(strikes, key=lambda s: abs(s - spot))

    ce_oi = pe_oi = ce_vol = pe_vol = 0.0
    ce_chg_oi = pe_chg_oi = 0.0
    atm_ce_ltp = atm_pe_ltp = atm_ce_iv = atm_pe_iv = 0.0
    max_ce_oi = max_pe_oi = -1.0
    max_ce_strike = max_pe_strike = atm

    for row in data:
        strike = float(row["strikePrice"])
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}
        coi = float(ce.get("openInterest") or 0.0)
        poi = float(pe.get("openInterest") or 0.0)
        ce_oi += coi
        pe_oi += poi
        ce_vol += float(ce.get("totalTradedVolume") or 0.0)
        pe_vol += float(pe.get("totalTradedVolume") or 0.0)
        ce_chg_oi += float(ce.get("changeinOpenInterest") or 0.0)
        pe_chg_oi += float(pe.get("changeinOpenInterest") or 0.0)
        if coi > max_ce_oi:
            max_ce_oi, max_ce_strike = coi, strike
        if poi > max_pe_oi:
            max_pe_oi, max_pe_strike = poi, strike
        if strike == atm:
            atm_ce_ltp = float(ce.get("lastPrice") or 0.0)
            atm_pe_ltp = float(pe.get("lastPrice") or 0.0)
            atm_ce_iv = float(ce.get("impliedVolatility") or 0.0)
            atm_pe_iv = float(pe.get("impliedVolatility") or 0.0)

    pcr_oi = pe_oi / ce_oi if ce_oi else 0.0
    pcr_vol = pe_vol / ce_vol if ce_vol else 0.0
    return {
        "spot": spot,
        "atm_strike": atm,
        "pcr_oi": pcr_oi,
        "pcr_vol": pcr_vol,
        "ce_oi": ce_oi,
        "pe_oi": pe_oi,
        "ce_chg_oi": ce_chg_oi,
        "pe_chg_oi": pe_chg_oi,
        "oi_imbalance": (pe_oi - ce_oi) / (pe_oi + ce_oi) if (pe_oi + ce_oi) else 0.0,
        "atm_ce_ltp": atm_ce_ltp,
        "atm_pe_ltp": atm_pe_ltp,
        "atm_ce_iv": atm_ce_iv,
        "atm_pe_iv": atm_pe_iv,
        "atm_iv_skew": atm_pe_iv - atm_ce_iv,
        "max_ce_oi_strike": max_ce_strike,
        "max_pe_oi_strike": max_pe_strike,
        "max_ce_oi_distance_pct": (max_ce_strike - spot) / spot * 100.0,
        "max_pe_oi_distance_pct": (max_pe_strike - spot) / spot * 100.0,
        "atm_premium_ratio": (atm_ce_ltp / atm_pe_ltp) if atm_pe_ltp else 0.0,
    }
