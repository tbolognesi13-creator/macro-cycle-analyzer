import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    from fredapi import Fred
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fredapi", "-q"])
    from fredapi import Fred

try:
    import yfinance as yf
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Macro Cycle Analyzer Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_FRED_API_KEY = "7b401890caa9dae74e8d7550ad49c69d"
START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
ZSCORE_WINDOW = 36

COLORS = {
    "expansion": "#26A69A",
    "slowdown": "#FFA726",
    "recession": "#EF5350",
    "recovery": "#42A5F5",
    "neutral": "#78909C",
    "gold": "#FFD700",
    "purple": "#AB47BC",
    "green2": "#66BB6A",
    "cyan": "#4FC3F7",
    "teal2": "#80CBC4",
    "orange2": "#FF7043",
    "violet2": "#CE93D8",
}

PHASE_COLORS = {
    "EXPANSION": COLORS["expansion"],
    "SLOWDOWN": COLORS["slowdown"],
    "RECESSION": COLORS["recession"],
    "RECOVERY": COLORS["recovery"],
    "NEUTRAL": COLORS["neutral"],
}


# ============================================================
# HELPERS
# ============================================================

def safe_last(series: pd.Series, default=np.nan):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else default


def safe_series(df: pd.DataFrame, col: str, index=None) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    if index is None:
        index = df.index
    return pd.Series(np.nan, index=index)


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().shape[0] <= periods:
        return pd.Series(np.nan, index=s.index)
    return s.pct_change(periods)


def robust_zscore(series: pd.Series, window: int = 36, min_periods: int = 12) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0, np.nan)


def normalize_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.index) == 0:
        return df
    df = df.copy()
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df[~idx.isna()]
    df.index = idx[~idx.isna()]
    df = df.sort_index()
    return df


def filter_timeframe(df: pd.DataFrame, years_label: str) -> pd.DataFrame:
    if df.empty:
        return df
    if years_label == "Full":
        return df.copy()

    years_map = {"3Y": 3, "5Y": 5, "10Y": 10, "15Y": 15}
    years = years_map.get(years_label)
    if years is None:
        return df.copy()

    cutoff = df.index.max() - pd.DateOffset(years=years)
    return df[df.index >= cutoff].copy()


def make_downloadable_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


# ============================================================
# DOWNLOAD DATI
# ============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def download_fred(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    fred = Fred(api_key=api_key)

    series = {
        "gdp": "GDPC1",
        "indprod": "INDPRO",
        "retail": "RSXFS",
        "durable": "DGORDER",
        "payrolls": "PAYEMS",
        "unemploy": "UNRATE",
        "jolts": "JTSJOL",
        "cpi": "CPIAUCSL",
        "core_cpi": "CPILFESL",
        "pce": "PCEPI",
        "core_pce": "PCEPILFE",
        "m2": "M2SL",
        "credit": "TOTALSL",
        "consumer_conf": "UMCSENT",
        "ism_pmi": "NAPM",
        "yield_10y": "DGS10",
        "yield_2y": "DGS2",
        "yield_3m": "DTB3",
        "hy_spread": "BAMLH0A0HYM2",
        "ig_spread": "BAMLC0A0CM",
        "infl_exp_5y": "T5YIE",
        "infl_exp_10y": "T10YIE",
        "infl_exp_5y5y": "T5YIFR",
        "fedfunds": "FEDFUNDS",
        "effr": "EFFR",
        "fomc_fedfunds_med": "FEDTARMD",
        "fomc_fedfunds_high": "FEDTARRH",
    }

    dfs = {}
    for name, code in series.items():
        try:
            s = fred.get_series(code, observation_start=start_date, observation_end=end_date)
            if s is None or len(s) == 0:
                dfs[name] = pd.Series(dtype=float)
            else:
                dfs[name] = pd.to_numeric(s, errors="coerce")
        except Exception:
            dfs[name] = pd.Series(dtype=float)

    df = pd.DataFrame(dfs)

    if len(df.index) == 0:
        df = pd.DataFrame(index=pd.date_range(start_date, end_date, freq="M"))
    else:
        df = normalize_monthly_index(df)

    for c in series.keys():
        if c not in df.columns:
            df[c] = np.nan

    df = df.resample("M").last().ffill()
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def download_market(start_date: str, end_date: str) -> pd.DataFrame:
    tickers = {
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "russell": "^RUT",
        "gold": "GC=F",
        "silver": "SI=F",
        "oil": "CL=F",
        "dxy": "DX-Y.NYB",
        "tnx": "^TNX",
        "vix": "^VIX",
        "tip": "TIP",
        "eem": "EEM",
        "xlf": "XLF",
        "xle": "XLE",
        "xlk": "XLK",
        "xlu": "XLU",
        "xlp": "XLP",
        "hyg": "HYG",
        "agg": "AGG",
        "bil": "BIL",
        "gld": "GLD",
        "tlt": "TLT",
        "xlv": "XLV",
        "xli": "XLI",
        "spy": "SPY",
        "iwm": "IWM",
    }

    try:
        raw = yf.download(
            list(tickers.values()),
            start=start_date,
            end=end_date,
            interval="1mo",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False
        )

        close_data = {}
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0)
            for name, ticker in tickers.items():
                try:
                    if ticker in lvl0:
                        close_data[name] = pd.to_numeric(raw[ticker]["Close"], errors="coerce")
                    else:
                        close_data[name] = pd.Series(dtype=float)
                except Exception:
                    close_data[name] = pd.Series(dtype=float)
        else:
            for name in tickers.keys():
                close_data[name] = pd.Series(dtype=float)

        df = pd.DataFrame(close_data)

        if len(df.index) == 0:
            df = pd.DataFrame(index=pd.date_range(start_date, end_date, freq="M"))
        else:
            df = normalize_monthly_index(df)
            df.index = df.index.to_period("M").to_timestamp("M")

        for c in tickers.keys():
            if c not in df.columns:
                df[c] = np.nan

        return df.ffill()

    except Exception:
        idx = pd.date_range(start_date, end_date, freq="M")
        df = pd.DataFrame(index=idx)
        for c in tickers.keys():
            df[c] = np.nan
        return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(fred_df: pd.DataFrame, mkt_df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=fred_df.index)
    aligned = mkt_df.reindex(fred_df.index, method="ffill")

    f["gdp_yoy"] = safe_pct_change(safe_series(fred_df, "gdp"), 4) * 100
    f["indprod_yoy"] = safe_pct_change(safe_series(fred_df, "indprod"), 12) * 100
    f["retail_yoy"] = safe_pct_change(safe_series(fred_df, "retail"), 12) * 100
    f["durable_yoy"] = safe_pct_change(safe_series(fred_df, "durable"), 12) * 100
    f["ism_pmi_level"] = safe_series(fred_df, "ism_pmi") - 50

    f["payrolls_mom"] = safe_series(fred_df, "payrolls").diff(3)
    f["unemploy_inv"] = -safe_series(fred_df, "unemploy")
    f["jolts_yoy"] = safe_pct_change(safe_series(fred_df, "jolts"), 12) * 100

    f["cpi_yoy"] = safe_pct_change(safe_series(fred_df, "cpi"), 12) * 100
    f["core_cpi_yoy"] = safe_pct_change(safe_series(fred_df, "core_cpi"), 12) * 100
    f["pce_yoy"] = safe_pct_change(safe_series(fred_df, "pce"), 12) * 100
    f["core_pce_yoy"] = safe_pct_change(safe_series(fred_df, "core_pce"), 12) * 100
    f["oil_yoy"] = safe_pct_change(safe_series(aligned, "oil"), 12) * 100

    f["yield_spread_10_2"] = safe_series(fred_df, "yield_10y") - safe_series(fred_df, "yield_2y")
    f["yield_spread_10_3m"] = safe_series(fred_df, "yield_10y") - safe_series(fred_df, "yield_3m")
    f["hy_spread_inv"] = -safe_series(fred_df, "hy_spread")
    f["ig_spread_inv"] = -safe_series(fred_df, "ig_spread")
    f["m2_yoy"] = safe_pct_change(safe_series(fred_df, "m2"), 12) * 100
    f["credit_yoy"] = safe_pct_change(safe_series(fred_df, "credit"), 12) * 100
    f["vix_inv"] = -safe_series(aligned, "vix")

    f["consumer_conf"] = safe_series(fred_df, "consumer_conf")
    f["sp500_6m"] = safe_pct_change(safe_series(aligned, "sp500"), 6) * 100
    f["sp500_12m"] = safe_pct_change(safe_series(aligned, "sp500"), 12) * 100
    f["gold_6m"] = safe_pct_change(safe_series(aligned, "gold"), 6) * 100
    f["dxy_6m"] = safe_pct_change(safe_series(aligned, "dxy"), 6) * 100

    f["infl_exp_5y_lvl"] = safe_series(fred_df, "infl_exp_5y")
    f["infl_exp_10y_lvl"] = safe_series(fred_df, "infl_exp_10y")
    f["infl_exp_5y5y_lvl"] = safe_series(fred_df, "infl_exp_5y5y")
    f["infl_exp_5y_chg_3m"] = safe_series(fred_df, "infl_exp_5y").diff(3)
    f["infl_exp_10y_chg_3m"] = safe_series(fred_df, "infl_exp_10y").diff(3)
    f["infl_exp_5y5y_chg_3m"] = safe_series(fred_df, "infl_exp_5y5y").diff(3)

    f["fedfunds_lvl"] = safe_series(fred_df, "fedfunds")
    f["effr_lvl"] = safe_series(fred_df, "effr")
    f["fomc_fedfunds_med_lvl"] = safe_series(fred_df, "fomc_fedfunds_med")
    f["fomc_fedfunds_high_lvl"] = safe_series(fred_df, "fomc_fedfunds_high")
    f["policy_gap_med"] = safe_series(fred_df, "fomc_fedfunds_med") - safe_series(fred_df, "fedfunds")
    f["policy_gap_high"] = safe_series(fred_df, "fomc_fedfunds_high") - safe_series(fred_df, "fedfunds")
    f["policy_gap_med_chg_3m"] = f["policy_gap_med"].diff(3)
    f["policy_gap_high_chg_3m"] = f["policy_gap_high"].diff(3)

    for col in f.columns:
        f[col] = robust_zscore(f[col], window=ZSCORE_WINDOW)

    return f


DIMENSION_WEIGHTS = {
    "growth": {
        "gdp_yoy": 0.22,
        "indprod_yoy": 0.18,
        "retail_yoy": 0.14,
        "durable_yoy": 0.10,
        "ism_pmi_level": 0.18,
        "sp500_6m": 0.09,
        "sp500_12m": 0.09,
    },
    "labor": {
        "payrolls_mom": 0.40,
        "unemploy_inv": 0.35,
        "jolts_yoy": 0.25,
    },
    "inflation": {
        "cpi_yoy": 0.20,
        "core_cpi_yoy": 0.18,
        "pce_yoy": 0.16,
        "core_pce_yoy": 0.12,
        "oil_yoy": 0.08,
        "infl_exp_5y_lvl": 0.08,
        "infl_exp_10y_lvl": 0.06,
        "infl_exp_5y5y_lvl": 0.06,
        "infl_exp_5y_chg_3m": 0.03,
        "infl_exp_5y5y_chg_3m": 0.03,
    },
    "financial": {
        "yield_spread_10_2": 0.14,
        "yield_spread_10_3m": 0.14,
        "hy_spread_inv": 0.14,
        "ig_spread_inv": 0.10,
        "m2_yoy": 0.08,
        "credit_yoy": 0.08,
        "vix_inv": 0.08,
        "policy_gap_med": -0.10,
        "policy_gap_high": -0.08,
        "policy_gap_med_chg_3m": -0.03,
        "policy_gap_high_chg_3m": -0.03,
    },
    "sentiment": {
        "consumer_conf": 0.50,
        "gold_6m": -0.20,
        "dxy_6m": -0.30,
    },
}

GLOBAL_WEIGHTS = {
    "growth": 0.30,
    "labor": 0.22,
    "inflation": -0.16,
    "financial": 0.20,
    "sentiment": 0.12,
}


def compute_scores(features: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=features.index)

    for dim, weights in DIMENSION_WEIGHTS.items():
        parts = []
        absw = []
        for k, w in weights.items():
            if k in features.columns:
                parts.append(features[k] * w)
                absw.append(abs(w))

        if len(parts) == 0:
            scores[dim] = np.nan
        else:
            agg = parts[0].copy()
            for p in parts[1:]:
                agg = agg.add(p, fill_value=np.nan)
            scores[dim] = agg / sum(absw)
            scores[dim] = scores[dim].rolling(3).mean()

    total = pd.Series(0.0, index=scores.index)
    total_w = 0.0
    for d, w in GLOBAL_WEIGHTS.items():
        if d in scores.columns:
            total = total.add(scores[d].fillna(0) * w, fill_value=0)
            total_w += abs(w)

    scores["global"] = total / total_w if total_w else np.nan
    scores["global_smooth"] = scores["global"].rolling(4).mean()
    scores["momentum"] = scores["global_smooth"].diff(2)
    return scores


def classify_phase(scores: pd.DataFrame) -> pd.Series:
    phases = []

    for idx in scores.index:
        row = scores.loc[idx]

        growth = float(row.get("growth", 0)) if pd.notna(row.get("growth", np.nan)) else 0
        labor = float(row.get("labor", 0)) if pd.notna(row.get("labor", np.nan)) else 0
        inflation = float(row.get("inflation", 0)) if pd.notna(row.get("inflation", np.nan)) else 0
        financial = float(row.get("financial", 0)) if pd.notna(row.get("financial", np.nan)) else 0
        momentum = float(row.get("momentum", 0)) if pd.notna(row.get("momentum", np.nan)) else 0

        growth_combo = 0.6 * growth + 0.4 * labor

        if growth_combo > 0.35 and inflation > 0.00 and financial >= -0.15:
            phase = "EXPANSION"
        elif growth_combo > 0.20 and momentum > 0 and inflation <= 0.10:
            phase = "RECOVERY"
        elif growth_combo <= -0.30 and financial < -0.20:
            phase = "RECESSION"
        elif growth_combo < 0 and momentum < -0.10:
            phase = "SLOWDOWN"
        elif growth_combo <= -0.35:
            phase = "RECESSION"
        else:
            phase = "NEUTRAL"

        if financial < -1.0 and labor < -0.5:
            phase = "RECESSION"

        phases.append(phase)

    return pd.Series(phases, index=scores.index)


def recession_probability(scores: pd.DataFrame) -> pd.Series:
    s = scores["global_smooth"].fillna(0)
    prob = 1 / (1 + np.exp(1.8 * s))
    return (prob * 100).round(1)


# ============================================================
# ANALOGHI STORICI
# ============================================================

def find_similar_historical_periods(scores: pd.DataFrame,
                                    phases: pd.Series,
                                    mkt_df: pd.DataFrame,
                                    top_n: int = 5,
                                    min_gap_months: int = 12) -> pd.DataFrame:
    compare_cols = [
        "growth", "labor", "inflation", "financial",
        "sentiment", "global_smooth", "momentum"
    ]

    df = scores.copy()
    df["phase"] = phases

    for c in compare_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df.dropna(subset=["global_smooth"]).copy()
    if len(df) < 24:
        return pd.DataFrame()

    current_idx = df.index[-1]
    current_vec = df.loc[current_idx, compare_cols].fillna(0.0).astype(float)

    eligible = df.iloc[:-min_gap_months].copy() if len(df) > min_gap_months else df.iloc[:-1].copy()
    if eligible.empty:
        return pd.DataFrame()

    rows = []
    spx = safe_series(mkt_df, "sp500")

    for idx, row in eligible.iterrows():
        vec = row[compare_cols].fillna(0.0).astype(float)
        dist = np.sqrt(((vec - current_vec) ** 2).sum())

        rec = {
            "date": idx,
            "distance": dist,
            "phase_then": phases.loc[idx] if idx in phases.index else np.nan,
        }

        if idx in spx.index:
            pos = spx.index.get_loc(idx)
            for m in [3, 6, 12]:
                try:
                    if isinstance(pos, slice):
                        rec[f"sp500_fwd_{m}m"] = np.nan
                    elif pos + m < len(spx):
                        p0 = spx.iloc[pos]
                        p1 = spx.iloc[pos + m]
                        rec[f"sp500_fwd_{m}m"] = ((p1 / p0) - 1) * 100 if pd.notna(p0) and pd.notna(p1) and p0 != 0 else np.nan
                    else:
                        rec[f"sp500_fwd_{m}m"] = np.nan
                except Exception:
                    rec[f"sp500_fwd_{m}m"] = np.nan

        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("distance").head(top_n)
    if out.empty:
        return out

    out["date"] = pd.to_datetime(out["date"])
    out["date_label"] = out["date"].dt.strftime("%Y-%m")
    return out


# ============================================================
# INTERPRETAZIONE MACRO
# ============================================================

def build_macro_interpretation(scores: pd.DataFrame,
                               fred_df: pd.DataFrame,
                               mkt_df: pd.DataFrame,
                               phase: str,
                               rec_prob: float) -> dict:
    out = {
        "macro_message": [],
        "asset_bias": [],
        "risk_flags": [],
        "summary": ""
    }

    def last(df, col, default=np.nan):
        if col not in df.columns:
            return default
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else default

    def chg(df, col, months=3, default=np.nan):
        if col not in df.columns:
            return default
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) <= months:
            return default
        return float(s.iloc[-1] - s.iloc[-1 - months])

    infl5 = last(fred_df, "infl_exp_5y")
    infl10 = last(fred_df, "infl_exp_10y")
    infl5y5y = last(fred_df, "infl_exp_5y5y")
    infl5_chg = chg(fred_df, "infl_exp_5y", 3)
    infl5y5y_chg = chg(fred_df, "infl_exp_5y5y", 3)

    fedfunds = last(fred_df, "fedfunds")
    effr = last(fred_df, "effr")
    fomc_med = last(fred_df, "fomc_fedfunds_med")

    policy_gap_med = fomc_med - fedfunds if pd.notna(fomc_med) and pd.notna(fedfunds) else np.nan
    vix_now = last(mkt_df, "vix", 20)

    momentum = float(scores["momentum"].dropna().iloc[-1]) if "momentum" in scores.columns and scores["momentum"].dropna().shape[0] else 0
    financial = float(scores["financial"].dropna().iloc[-1]) if "financial" in scores.columns and scores["financial"].dropna().shape[0] else 0

    if phase == "EXPANSION":
        out["macro_message"].append("Il quadro resta coerente con una fase di espansione: crescita e mercato del lavoro tengono.")
    elif phase == "RECOVERY":
        out["macro_message"].append("Il quadro suggerisce una fase di recovery: il ciclo migliora ma non è ancora una espansione piena.")
    elif phase == "SLOWDOWN":
        out["macro_message"].append("Il quadro è da rallentamento: crescita e momentum si stanno indebolendo.")
    elif phase == "RECESSION":
        out["macro_message"].append("Il modello è coerente con una fase recessiva o molto vicina a una recessione.")
    else:
        out["macro_message"].append("Il quadro è intermedio: segnali misti, senza regime dominante.")

    if pd.notna(infl5_chg) and pd.notna(infl5y5y_chg):
        if infl5_chg > 0.20 and infl5y5y_chg > 0.10:
            out["macro_message"].append(f"Le aspettative di inflazione stanno risalendo: 5Y {infl5:.2f}%, 10Y {infl10:.2f}%, 5Y5Y {infl5y5y:.2f}%.")
            out["asset_bias"].append("Bias positivo per TIPS, oro e asset reali; meno favorevoli i Treasury nominali lunghi.")
        elif infl5_chg < -0.20 and infl5y5y_chg < -0.10:
            out["macro_message"].append(f"Le aspettative di inflazione stanno scendendo: 5Y {infl5:.2f}%, 10Y {infl10:.2f}%, 5Y5Y {infl5y5y:.2f}%.")
            out["asset_bias"].append("Bias più favorevole ai Treasury nominali lunghi; meno necessario sovrappesare TIPS e oro.")
        else:
            out["macro_message"].append(f"Le aspettative di inflazione sono relativamente stabili: 5Y {infl5:.2f}%, 10Y {infl10:.2f}%, 5Y5Y {infl5y5y:.2f}%.")

    if pd.notna(policy_gap_med):
        if policy_gap_med > 0.50:
            out["macro_message"].append(f"Le aspettative sui tassi restano restrittive rispetto al livello corrente: gap mediano {policy_gap_med:+.2f}%, Fed Funds {fedfunds:.2f}%, EFFR {effr:.2f}%.")
            out["asset_bias"].append("Meglio prudenza sulla duration lunga; più interessanti cash, T-bill e qualità difensiva.")
        elif policy_gap_med < -0.25:
            out["macro_message"].append(f"Le aspettative sui tassi appaiono più accomodanti del livello corrente: gap mediano {policy_gap_med:+.2f}%.")
            out["asset_bias"].append("Bias più favorevole a Treasury medi e lunghi e agli asset sensibili al calo dei tassi.")
        else:
            out["macro_message"].append(f"Le aspettative sui tassi sono relativamente vicine al livello corrente: gap mediano {policy_gap_med:+.2f}%.")

    if pd.notna(policy_gap_med) and pd.notna(infl5_chg):
        if policy_gap_med > 0.50 and infl5_chg > 0.15:
            out["asset_bias"].append("Combinazione sfavorevole per long duration nominale: meglio breve termine, TIPS, oro e difensivi.")
            out["risk_flags"].append("Rischio di repricing obbligazionario se il mercato rivede al rialzo tassi reali e inflazione attesa.")
        elif policy_gap_med < 0 and infl5_chg < 0:
            out["asset_bias"].append("Scenario relativamente favorevole per Treasury nominali lunghi e duration di qualità.")
        elif policy_gap_med > 0 and infl5_chg < 0:
            out["asset_bias"].append("Meglio approccio barbell: cash/T-bill da una parte, duration selettiva dall'altra.")
        elif policy_gap_med < 0 and infl5_chg > 0:
            out["asset_bias"].append("Scenario da monitorare: tagli attesi ma inflazione ancora appiccicosa, quindi TIPS più interessanti dei nominali lunghi.")

    if financial < -0.5 or vix_now > 25:
        out["risk_flags"].append("Stress finanziario o volatilità elevata: servono disciplina difensiva e attenzione al rischio.")
    if rec_prob > 55:
        out["risk_flags"].append(f"Probabilità di recessione elevata ({rec_prob:.1f}%).")
    if momentum < -0.10:
        out["risk_flags"].append("Momentum macro in deterioramento.")

    summary_parts = []
    if out["asset_bias"]:
        summary_parts.append(" | ".join(out["asset_bias"][:3]))
    if out["risk_flags"]:
        summary_parts.append("Rischi: " + " | ".join(out["risk_flags"][:2]))
    out["summary"] = " ".join(summary_parts) if summary_parts else "Quadro senza bias tattico dominante."

    return out


# ============================================================
# INVESTMENT ENGINE
# ============================================================

def build_investment_picks(scores: pd.DataFrame,
                           phase: str,
                           rec_prob: float,
                           mkt_df: pd.DataFrame,
                           fred_df: pd.DataFrame) -> list:
    last = scores.iloc[-1]

    g = float(last.get("growth", 0)) if pd.notna(last.get("growth", np.nan)) else 0
    l = float(last.get("labor", 0)) if pd.notna(last.get("labor", np.nan)) else 0
    inf = float(last.get("inflation", 0)) if pd.notna(last.get("inflation", np.nan)) else 0
    fin = float(last.get("financial", 0)) if pd.notna(last.get("financial", np.nan)) else 0
    sen = float(last.get("sentiment", 0)) if pd.notna(last.get("sentiment", np.nan)) else 0
    mom = float(last.get("momentum", 0)) if pd.notna(last.get("momentum", np.nan)) else 0

    def safe_perf(col, periods=6):
        if col in mkt_df.columns:
            s = pd.to_numeric(mkt_df[col], errors="coerce").dropna()
            if len(s) > periods:
                return float(s.pct_change(periods).iloc[-1] * 100)
        return 0.0

    gold_perf = safe_perf("gold", 6)
    cpi_series = safe_series(fred_df, "cpi")
    cpi_now = float(cpi_series.pct_change(12).dropna().iloc[-1] * 100) if cpi_series.dropna().shape[0] > 12 else 3.0

    high_rec = rec_prob > 50
    very_high = rec_prob > 65
    neg_mom = mom < -0.05
    gold_hot = gold_perf > 10

    infl5_chg = np.nan
    if "infl_exp_5y" in fred_df.columns:
        s = pd.to_numeric(fred_df["infl_exp_5y"], errors="coerce").dropna()
        if len(s) > 3:
            infl5_chg = float(s.iloc[-1] - s.iloc[-4])

    policy_gap_med = np.nan
    if "fomc_fedfunds_med" in fred_df.columns and "fedfunds" in fred_df.columns:
        ff = pd.to_numeric(fred_df["fedfunds"], errors="coerce").dropna()
        fm = pd.to_numeric(fred_df["fomc_fedfunds_med"], errors="coerce").dropna()
        if len(ff) and len(fm):
            policy_gap_med = float(fm.iloc[-1] - ff.iloc[-1])

    def pick(ticker, name, weight, signal, rationale, risk):
        return {
            "ticker": ticker,
            "name": name,
            "weight": weight,
            "signal": signal,
            "rationale": rationale,
            "risk": risk
        }

    if phase == "RECESSION" or very_high:
        hawkish = pd.notna(policy_gap_med) and policy_gap_med > 0.50
        infl_down = pd.notna(infl5_chg) and infl5_chg < 0

        if infl_down and not hawkish:
            return [
                pick("TLT", "Treasury 20Y+", 30, "FORTE ACQUISTO", "Recessione con inflazione attesa in discesa: contesto favorevole alla duration lunga.", "Volatilità se il mercato rivaluta il percorso dei tassi."),
                pick("GLD", "Gold ETF", 22, "ACQUISTO", "Hedge difensivo e diversificatore.", "Può sottoperformare i bond in disinflazione forte."),
                pick("BIL", "T-Bill 1-3M", 18, "ACQUISTO", "Cuscinetto liquido per volatilità e ribilanciamento.", "Carry in calo con Fed più accomodante."),
                pick("XLP", "Consumer Staples", 16, "ACQUISTO", "Difensivo classico nei drawdown macro.", "Valutazioni talvolta già tirate."),
                pick("XLV", "Healthcare", 14, "ACQUISTO", "Minore ciclicità e maggiore stabilità.", "Rischio regolatorio."),
            ]
        return [
            pick("GLD", "Gold ETF", 28, "FORTE ACQUISTO", "Recessione con inflazione attesa non del tutto rientrata: oro preferibile come hedge reale.", "Meglio ingresso graduale."),
            pick("TIP", "TIPS", 22, "ACQUISTO", "Se la recessione convive con inflazione attesa appiccicosa, TIPS meglio dei nominali lunghi.", "Se arriva vera disinflazione, TLT può fare meglio."),
            pick("BIL", "T-Bill 1-3M", 20, "ACQUISTO", "Protezione tattica e flessibilità.", "Carry in riduzione se la Fed taglia."),
            pick("XLP", "Consumer Staples", 16, "ACQUISTO", "Difensivo coerente con contrazione.", "Upside contenuto."),
            pick("XLV", "Healthcare", 14, "ACQUISTO", "Settore difensivo robusto.", "Rischio regolatorio."),
        ]

    if phase == "SLOWDOWN" or (high_rec and neg_mom):
        hawkish = pd.notna(policy_gap_med) and policy_gap_med > 0.50
        dovish = pd.notna(policy_gap_med) and policy_gap_med < 0
        infl_up = pd.notna(infl5_chg) and infl5_chg > 0.15
        infl_down = pd.notna(infl5_chg) and infl5_chg < 0

        if hawkish and infl_up:
            return [
                pick("BIL", "T-Bill 1-3M", 28, "FORTE ACQUISTO", "Aspettative tassi restrittive: cash remunerato più efficiente della duration lunga.", "Rendimento in calo se il mercato anticipa forti tagli."),
                pick("TIP", "TIPS", 24, "FORTE ACQUISTO", "Aspettative di inflazione in salita: meglio TIPS dei Treasury nominali lunghi.", "Se l'inflazione attesa si raffredda, il vantaggio relativo cala."),
                pick("GLD", "Gold ETF", 20, "ACQUISTO", "Oro favorito come hedge contro inflazione attesa persistente e incertezza macro.", "Meglio ingresso graduale."),
                pick("XLP", "Consumer Staples", 16, "ACQUISTO", "Settore difensivo coerente con rallentamento.", "Upside limitato in risk-on improvviso."),
                pick("XLU", "Utilities", 12, "ACQUISTO", "Profilo difensivo e cash-flow resilienti.", "Sensibile ai tassi lunghi."),
            ]

        if dovish and infl_down:
            return [
                pick("TLT", "Treasury 20Y+", 28, "FORTE ACQUISTO", "Tassi attesi in calo e inflazione attesa in raffreddamento: contesto migliore per duration lunga.", "Volatilità se il mercato rivaluta il percorso Fed."),
                pick("GLD", "Gold ETF", 20, "ACQUISTO", "Diversificatore difensivo in quadro fragile.", "Può sottoperformare in disinflazione accelerata."),
                pick("BIL", "T-Bill 1-3M", 18, "ACQUISTO", "Mantiene flessibilità tattica.", "Carry meno interessante se la Fed taglia rapidamente."),
                pick("XLP", "Consumer Staples", 18, "ACQUISTO", "Difensivo classico da slowdown.", "Partecipazione limitata a rimbalzi forti."),
                pick("XLV", "Healthcare", 16, "ACQUISTO", "Resilienza degli utili e beta più basso.", "Rischio regolatorio."),
            ]

        gold_w = 24 if gold_hot else 20
        bil_w = 24
        xlp_w = 18
        xlu_w = 14
        tip_w = 100 - gold_w - bil_w - xlp_w - xlu_w
        return [
            pick("GLD", "Gold ETF", gold_w, "ACQUISTO", "Slowdown con rischio recessivo: oro utile come hedge macro.", "Meglio accumulo graduale."),
            pick("BIL", "T-Bill 1-3M", bil_w, "FORTE ACQUISTO", "Breve termine efficiente in attesa di chiarezza sui tassi.", "Carry in discesa se tagli aggressivi."),
            pick("XLP", "Consumer Staples", xlp_w, "ACQUISTO", "Settore difensivo coerente con rallentamento.", "Upside contenuto."),
            pick("XLU", "Utilities", xlu_w, "ACQUISTO", "Difensivo con stabilità relativa.", "Soffre rialzo dei tassi lunghi."),
            pick("TIP", "TIPS", tip_w, "NEUTRO/ACQUISTO", "Buon compromesso se il rallentamento convive con inflazione non del tutto rientrata.", "Se l'inflazione cala rapidamente, TLT può fare meglio."),
        ]

    if phase == "EXPANSION":
        return [
            pick("XLK", "Technology", 30, "FORTE ACQUISTO", "Espansione con growth e condizioni finanziarie favorevoli.", "Valutazioni elevate."),
            pick("XLF", "Financials", 20, "ACQUISTO", "Beneficia di credito in crescita e attività economica.", "Molto ciclico."),
            pick("EEM", "Emergenti", 20, "ACQUISTO", "Leva sul ciclo globale.", "Rischio geopolitico e valutario."),
            pick("XLE", "Energy", 15, "ACQUISTO", "Inflazione e domanda ciclica supportano energia.", "Volatilità commodity."),
            pick("GLD", "Gold ETF", 15, "NEUTRO", "Diversificazione.", "Opportunity cost più alto."),
        ]

    if phase == "RECOVERY":
        return [
            pick("IWM", "Small Cap", 25, "FORTE ACQUISTO", "Recovery con growth in miglioramento.", "Volatilità elevata."),
            pick("XLI", "Industrials", 20, "ACQUISTO", "Beneficia della riaccelerazione ciclica.", "Sensibile a delusioni sulla crescita."),
            pick("EEM", "Emergenti", 20, "ACQUISTO", "Scenario favorevole in recupero del ciclo globale.", "Rischio geopolitico."),
            pick("XLE", "Energy", 20, "ACQUISTO", "Reflazione e ripresa supportano il comparto.", "Volatilità elevata."),
            pick("GLD", "Gold ETF", 15, "NEUTRO", "Difensivo residuo.", "Può sottoperformare in risk-on forte."),
        ]

    return [
        pick("SPY", "S&P 500", 30, "NEUTRO/ACQUISTO", "Esposizione core al mercato USA in fase neutrale.", "Sensibile a peggioramento del ciclo."),
        pick("GLD", "Gold ETF", 20, "ACQUISTO", "Diversificatore strutturale.", "Assenza di carry."),
        pick("BIL", "T-Bill 1-3M", 20, "ACQUISTO", "Buffer di liquidità.", "Rendimento scende con i tagli."),
        pick("XLP", "Consumer Staples", 15, "NEUTRO/ACQUISTO", "Difensivo equilibrato.", "Poco upside in rally ampio."),
        pick("AGG", "Bond Aggregate", 15, "NEUTRO", "Diversificazione obbligazionaria.", "Duration media."),
    ]


# ============================================================
# UI
# ============================================================

st.title("Macro Cycle Analyzer Pro")
st.caption("Dashboard macroeconomica interattiva con investment engine, aspettative inflazione/tassi e analoghi storici")

with st.sidebar:
    st.header("Impostazioni")
    fred_api_key = st.text_input("FRED API Key", value=DEFAULT_FRED_API_KEY, type="password")
    timeframe = st.selectbox("Timeframe", ["3Y", "5Y", "10Y", "15Y", "Full"], index=2)
    show_raw = st.checkbox("Mostra dati grezzi", value=False)
    run_button = st.button("Aggiorna analisi", type="primary", use_container_width=True)

if run_button or "loaded_once" not in st.session_state:
    st.session_state.loaded_once = True

    with st.spinner("Scaricamento dati macro e di mercato..."):
        fred_df = download_fred(fred_api_key, START_DATE, END_DATE)
        mkt_df = download_market(START_DATE, END_DATE)
        features = build_features(fred_df, mkt_df)
        scores = compute_scores(features)
        phases = classify_phase(scores)
        rec_prob = recession_probability(scores)
        similar_df_full = find_similar_historical_periods(scores, phases, mkt_df, top_n=5, min_gap_months=12)

    df_all = pd.concat([scores, phases.rename("phase"), rec_prob.rename("recession_prob")], axis=1)

    fred_view = filter_timeframe(fred_df, timeframe)
    mkt_view = filter_timeframe(mkt_df, timeframe)
    scores_view = filter_timeframe(df_all, timeframe)

    # FIX DEFINITIVO: phase non è dentro scores
    current_phase = phases.iloc[-1]
    current_rec = float(rec_prob.iloc[-1]) if len(rec_prob) else np.nan
    current_score = safe_last(scores["global_smooth"], 0.0)
    current_mom = safe_last(scores["momentum"], 0.0)
    phase_color = PHASE_COLORS.get(current_phase, COLORS["neutral"])

    picks = build_investment_picks(scores, current_phase, current_rec, mkt_df, fred_df)
    interpretation = build_macro_interpretation(scores, fred_df, mkt_df, current_phase, current_rec)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fase macro", current_phase)
    c2.metric("Prob. recessione", f"{current_rec:.1f}%")
    c3.metric("Score globale", f"{current_score:+.2f}")
    c4.metric("Momentum", f"{current_mom:+.2f}")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(
            x=scores_view.index,
            y=scores_view["global_smooth"],
            mode="lines",
            name="Global Smooth",
            line=dict(color=phase_color, width=3)
        ))
        fig_main.add_trace(go.Scatter(
            x=scores_view.index,
            y=scores_view["global"],
            mode="lines",
            name="Global Raw",
            line=dict(width=1),
            opacity=0.35
        ))
        fig_main.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_main.update_layout(
            title="Score Macroeconomico Globale",
            height=420,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_main, use_container_width=True)

    with right:
        fig_rec = go.Figure()
        fig_rec.add_trace(go.Scatter(
            x=scores_view.index,
            y=scores_view["recession_prob"],
            fill="tozeroy",
            mode="lines",
            name="Recession Probability",
            line=dict(color=COLORS["recession"], width=3)
        ))
        fig_rec.add_hline(y=50, line_dash="dash", line_color=COLORS["slowdown"])
        fig_rec.update_layout(
            title="Probabilità di Recessione",
            height=420,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(range=[0, 100], ticksuffix="%")
        )
        st.plotly_chart(fig_rec, use_container_width=True)

    st.divider()

    c5, c6 = st.columns([2, 1])

    with c5:
        fig_dims = go.Figure()
        dim_map = {
            "growth": COLORS["expansion"],
            "labor": COLORS["green2"],
            "inflation": COLORS["slowdown"],
            "financial": COLORS["recovery"],
            "sentiment": COLORS["purple"],
        }
        for d, col in dim_map.items():
            if d in scores_view.columns:
                fig_dims.add_trace(go.Scatter(
                    x=scores_view.index,
                    y=scores_view[d],
                    mode="lines",
                    name=d.capitalize(),
                    line=dict(color=col, width=2)
                ))
        fig_dims.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_dims.update_layout(
            title="Score per Dimensione",
            height=420,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_dims, use_container_width=True)

    with c6:
        last_scores = pd.DataFrame({
            "Dimension": ["Growth", "Labor", "Inflation", "Financial", "Sentiment"],
            "Value": [
                safe_last(scores["growth"], 0),
                safe_last(scores["labor"], 0),
                safe_last(scores["inflation"], 0),
                safe_last(scores["financial"], 0),
                safe_last(scores["sentiment"], 0),
            ]
        })
        fig_bar = px.bar(
            last_scores,
            x="Dimension",
            y="Value",
            template="plotly_dark",
            title="Ultimo profilo dimensionale"
        )
        fig_bar.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    c7, c8 = st.columns([2, 1])

    with c7:
        recent_mkt = mkt_view.iloc[-24:].copy()
        fig_mkt = go.Figure()
        for asset, color in {
            "sp500": COLORS["expansion"],
            "gold": COLORS["gold"],
            "dxy": COLORS["cyan"],
            "oil": COLORS["orange2"],
            "vix": COLORS["violet2"],
        }.items():
            if asset in recent_mkt.columns:
                s = pd.to_numeric(recent_mkt[asset], errors="coerce").dropna()
                if len(s) > 0:
                    norm = (s / s.iloc[0]) * 100
                    fig_mkt.add_trace(go.Scatter(
                        x=norm.index,
                        y=norm.values,
                        mode="lines",
                        name=asset.upper(),
                        line=dict(color=color, width=2)
                    ))
        fig_mkt.add_hline(y=100, line_dash="dash", line_color="gray")
        fig_mkt.update_layout(
            title="Performance Asset (base 100)",
            height=420,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_mkt, use_container_width=True)

    with c8:
        spread_10_2 = safe_series(fred_view, "yield_10y") - safe_series(fred_view, "yield_2y")
        spread_10_3m = safe_series(fred_view, "yield_10y") - safe_series(fred_view, "yield_3m")

        fig_yield = go.Figure()
        fig_yield.add_trace(go.Scatter(
            x=spread_10_2.index, y=spread_10_2,
            mode="lines", name="10Y-2Y",
            line=dict(color=COLORS["cyan"], width=2)
        ))
        fig_yield.add_trace(go.Scatter(
            x=spread_10_3m.index, y=spread_10_3m,
            mode="lines", name="10Y-3M",
            line=dict(color=COLORS["teal2"], width=2)
        ))
        fig_yield.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_yield.update_layout(
            title="Yield Curve Spread",
            height=420,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_yield, use_container_width=True)

    st.divider()

    c9, c10 = st.columns([1, 1])

    with c9:
        infl_exp_df = pd.DataFrame(index=fred_view.index)
        infl_exp_df["5Y"] = safe_series(fred_view, "infl_exp_5y")
        infl_exp_df["10Y"] = safe_series(fred_view, "infl_exp_10y")
        infl_exp_df["5Y5Y"] = safe_series(fred_view, "infl_exp_5y5y")

        fig_infl = go.Figure()
        for col, color in [("5Y", COLORS["gold"]), ("10Y", COLORS["cyan"]), ("5Y5Y", COLORS["purple"])]:
            fig_infl.add_trace(go.Scatter(
                x=infl_exp_df.index,
                y=infl_exp_df[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=2)
            ))
        fig_infl.update_layout(
            title="Aspettative di Inflazione",
            height=400,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(ticksuffix="%")
        )
        st.plotly_chart(fig_infl, use_container_width=True)

    with c10:
        rates_df = pd.DataFrame(index=fred_view.index)
        rates_df["FedFunds"] = safe_series(fred_view, "fedfunds")
        rates_df["EFFR"] = safe_series(fred_view, "effr")
        rates_df["FOMC Median"] = safe_series(fred_view, "fomc_fedfunds_med")

        fig_rates = go.Figure()
        for col, color in [("FedFunds", COLORS["cyan"]), ("EFFR", COLORS["green2"]), ("FOMC Median", COLORS["slowdown"])]:
            fig_rates.add_trace(go.Scatter(
                x=rates_df.index,
                y=rates_df[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=2)
            ))
        fig_rates.update_layout(
            title="Tassi Correnti e Aspettative FOMC",
            height=400,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(ticksuffix="%")
        )
        st.plotly_chart(fig_rates, use_container_width=True)

    st.divider()

    st.subheader("Interpretazione finale")

    if interpretation["macro_message"]:
        st.markdown("**Quadro macro**")
        for msg in interpretation["macro_message"]:
            st.write(f"• {msg}")

    if interpretation["asset_bias"]:
        st.markdown("**Implicazioni asset**")
        for msg in interpretation["asset_bias"]:
            st.write(f"• {msg}")

    if interpretation["risk_flags"]:
        st.markdown("**Rischi da monitorare**")
        for msg in interpretation["risk_flags"]:
            st.write(f"• {msg}")

    st.info(interpretation["summary"])

    st.divider()

    st.subheader("Investment Engine — Top Picks")
    picks_df = pd.DataFrame(picks)
    st.dataframe(
        picks_df[["ticker", "name", "weight", "signal", "rationale", "risk"]],
        use_container_width=True,
        hide_index=True
    )

    fig_alloc = px.pie(
        picks_df,
        names="ticker",
        values="weight",
        title=f"Allocazione dinamica suggerita — {current_phase}",
        template="plotly_dark"
    )
    fig_alloc.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.divider()

    st.subheader("Fasi storiche simili")
    if similar_df_full.empty:
        st.warning("Nessun confronto storico disponibile.")
    else:
        st.dataframe(
            similar_df_full[["date_label", "phase_then", "distance", "sp500_fwd_3m", "sp500_fwd_6m", "sp500_fwd_12m"]].round(2),
            use_container_width=True,
            hide_index=True
        )

        sim_plot = similar_df_full.copy()
        sim_plot["label"] = sim_plot["date_label"] + " | " + sim_plot["phase_then"].astype(str)

        fig_sim = go.Figure()
        for col, name in [
            ("sp500_fwd_3m", "S&P +3M"),
            ("sp500_fwd_6m", "S&P +6M"),
            ("sp500_fwd_12m", "S&P +12M"),
        ]:
            fig_sim.add_trace(go.Bar(
                x=sim_plot["label"],
                y=sim_plot[col],
                name=name
            ))
        fig_sim.update_layout(
            barmode="group",
            title="Performance forward degli episodi storici simili",
            height=420,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()

    st.subheader("Tabella score completa")
    st.dataframe(scores_view.round(3), use_container_width=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Scarica scores CSV",
            data=make_downloadable_csv(scores_view),
            file_name="macro_scores.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d2:
        st.download_button(
            "Scarica market CSV",
            data=make_downloadable_csv(mkt_view),
            file_name="market_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d3:
        st.download_button(
            "Scarica fred CSV",
            data=make_downloadable_csv(fred_view),
            file_name="fred_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d4:
        st.download_button(
            "Scarica analoghi storici CSV",
            data=make_downloadable_csv(similar_df_full) if not similar_df_full.empty else b"",
            file_name="similar_periods.csv",
            mime="text/csv",
            use_container_width=True
        )

    if show_raw:
        st.divider()
        st.subheader("Dati grezzi FRED")
        st.dataframe(fred_view, use_container_width=True)

        st.subheader("Dati grezzi Market")
        st.dataframe(mkt_view, use_container_width=True)