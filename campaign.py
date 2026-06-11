from __future__ import annotations

from datetime import datetime, date, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _as_date(value: str) -> date:
    return datetime.fromisoformat(str(value)).date()


def _as_time(value: str) -> time:
    hh, mm = str(value).strip().split(":")
    hh_i = int(hh)
    mm_i = int(mm)
    if not (0 <= hh_i <= 23 and 0 <= mm_i <= 59):
        raise ValueError("hora fora do intervalo")
    return time(hour=hh_i, minute=mm_i)


def validate_night_params(start_utc: str, end_utc: str, min_duration_min: int) -> List[str]:
    errors: List[str] = []
    for label, value in [("início", start_utc), ("fim", end_utc)]:
        try:
            _as_time(value)
        except Exception:
            errors.append(f"Hora de {label} da noite inválida. Use HH:MM.")
    if int(min_duration_min) <= 0:
        errors.append("A duração mínima da janela deve ser maior que zero.")
    return errors


def build_night_windows(data_inicio: str, data_fim: str, start_utc: str = "21:00", end_utc: str = "08:00") -> List[Dict[str, Any]]:
    d0 = _as_date(data_inicio)
    d1 = _as_date(data_fim)
    t0 = _as_time(start_utc)
    t1 = _as_time(end_utc)
    windows: List[Dict[str, Any]] = []
    d = d0
    while d <= d1:
        start = datetime.combine(d, t0)
        end = datetime.combine(d, t1)
        if end <= start:
            end += timedelta(days=1)
        windows.append({"Noite_UTC": d.isoformat(), "inicio": start, "fim": end})
        d += timedelta(days=1)
    return windows


def mpc_query_dates_for_nights(data_inicio: str, data_fim: str) -> Tuple[str, str, str]:
    """Retorna datas/hora para consultar MPC cobrindo noites 21-08.

    Usa 12:00 UTC como referência para incluir a noite inicial e a manhã após a última noite.
    """
    end = _as_date(data_fim) + timedelta(days=1)
    return data_inicio, end.isoformat(), "12:00"


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lower.get(c.lower())
        if hit is not None:
            return hit
    return None


def _numeric(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _assign_night(epoch: Any, windows: List[Dict[str, Any]]) -> Optional[str]:
    if pd.isna(epoch):
        return None
    ts = pd.Timestamp(epoch).to_pydatetime().replace(tzinfo=None)
    for w in windows:
        if w["inicio"] <= ts <= w["fim"]:
            return str(w["Noite_UTC"])
    return None


def filter_observable_nights(df_mpc: pd.DataFrame, cfg: Any, start_utc: str, end_utc: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    audit = {"linhas_entrada": int(len(df_mpc)) if df_mpc is not None else 0, "linhas_saida": 0, "avisos": []}
    if df_mpc is None or df_mpc.empty:
        return pd.DataFrame(), {**audit, "avisos": ["df_mpc vazio."]}

    df = df_mpc.copy()
    epoch_col = _find_col(df, ["epoch", "Epoch", "Date", "date", "datetime", "Time", "time"])
    if epoch_col is None:
        return pd.DataFrame(), {**audit, "avisos": ["Sem coluna de época; não foi possível separar por noite."]}

    df["epoch"] = pd.to_datetime(df[epoch_col], errors="coerce")
    windows = build_night_windows(cfg.data_inicio, cfg.data_fim, start_utc, end_utc)
    df["Noite_UTC"] = df["epoch"].map(lambda x: _assign_night(x, windows))
    df = df[df["Noite_UTC"].notna()].copy()

    colmap = {
        "V": ["V", "mag", "Mag", "Vmag", "Magnitude"],
        "Alt": ["Alt", "alt", "Altitude", "EL", "El", "elev", "Elevation"],
        "SunAlt": ["SunAlt", "sunAlt", "SolAlt", "SunEl", "Sun_EL", "Sun_elev", "SunElevation", "Sun altitude"],
        "alpha": ["alpha", "Alpha", "phase", "Phase", "Phase_angle", "PhaseAngle", "Phase angle"],
        "dRA": ["dRA", "RA_rate", "ra_rate", "RA motion", "RA_motion", "dRA/dt"],
        "dDec": ["dDec", "Dec_rate", "dec_rate", "Dec motion", "Dec_motion", "dDec/dt"],
        "mu_total": ["mu_total", "mu", "pm", "ProperMotion", "Proper motion", "proper_motion", "Sky motion", "sky_motion"],
    }
    for target, names in colmap.items():
        col = _find_col(df, names)
        if col is not None:
            df[target] = _numeric(df[col])

    if "mu_total" not in df.columns:
        if "dRA" in df.columns and "dDec" in df.columns:
            df["mu_total"] = np.sqrt(df["dRA"].fillna(0) ** 2 + df["dDec"].fillna(0) ** 2)
        else:
            df["mu_total"] = np.nan

    if "V" in df.columns:
        df = df[df["V"].notna() & (df["V"] <= float(cfg.V_MAX))]
    else:
        audit["avisos"].append("Sem coluna V/magnitude; filtro de magnitude não aplicado.")

    if "Alt" in df.columns:
        df = df[df["Alt"].notna() & (df["Alt"] >= float(cfg.ALT_MIN)) & (df["Alt"] <= float(cfg.ALT_MAX))]
    else:
        audit["avisos"].append("Sem coluna Alt/Altitude; filtro de altura não aplicado.")

    if getattr(cfg, "SOL_ALT_MAX", None) is not None:
        if "SunAlt" in df.columns:
            df = df[df["SunAlt"].notna() & (df["SunAlt"] <= float(cfg.SOL_ALT_MAX))]
        else:
            audit["avisos"].append("SOL_ALT_MAX definido, mas sem coluna SunAlt.")

    audit["linhas_saida"] = int(len(df))
    return df.reset_index(drop=True), audit


def _first(g: pd.DataFrame, col: str) -> Any:
    if col not in g.columns:
        return np.nan
    s = g[col].dropna()
    return s.iloc[0] if len(s) else np.nan


def _last(g: pd.DataFrame, col: str) -> Any:
    if col not in g.columns:
        return np.nan
    s = g[col].dropna()
    return s.iloc[-1] if len(s) else np.nan


def _quality(row: Dict[str, Any], cfg: Any, min_duration_min: int) -> str:
    dur = float(row.get("Duracao_h", 0.0)) * 60.0
    vmin = row.get("V_min", np.nan)
    altmax = row.get("ALT_max", np.nan)
    mu = row.get("mu_med", np.nan)
    vel_ok = pd.isna(mu) or float(mu) <= float(cfg.LIMIAR_RAPIDO)
    if dur >= max(90, min_duration_min) and pd.notna(vmin) and float(vmin) <= min(18.5, float(cfg.V_MAX)) and pd.notna(altmax) and float(altmax) >= max(40, float(cfg.ALT_MIN)) and vel_ok:
        return "Boa"
    if dur >= min_duration_min and (pd.isna(vmin) or float(vmin) <= float(cfg.V_MAX)) and (pd.isna(altmax) or float(altmax) >= float(cfg.ALT_MIN)):
        return "Regular"
    return "Ruim"


def summarize_by_night(df_obs: pd.DataFrame, cfg: Any, min_duration_min: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    audit = {"objetos_noites": 0, "avisos": []}
    if df_obs is None or df_obs.empty:
        return pd.DataFrame(), {**audit, "avisos": ["df_obs vazio."]}

    rows: List[Dict[str, Any]] = []
    for (obj, night), g in df_obs.sort_values("epoch").groupby(["Nome_limpo", "Noite_UTC"]):
        n = int(len(g))
        row = {
            "Nome_limpo": str(obj), "Objeto": str(obj), "Noite_UTC": str(night),
            "Inicio_janela_UTC": pd.to_datetime(g["epoch"]).min(),
            "Fim_janela_UTC": pd.to_datetime(g["epoch"]).max(),
            "Duracao_h": round((n * int(cfg.step_min)) / 60.0, 3), "n_epocas": n,
            "V_inicio": _first(g, "V"), "V_fim": _last(g, "V"), "V_min": _numeric(g["V"]).min() if "V" in g else np.nan, "V_med": _numeric(g["V"]).median() if "V" in g else np.nan,
            "ALT_inicio": _first(g, "Alt"), "ALT_fim": _last(g, "Alt"), "ALT_max": _numeric(g["Alt"]).max() if "Alt" in g else np.nan, "ALT_med": _numeric(g["Alt"]).median() if "Alt" in g else np.nan,
            "alpha_inicio": _first(g, "alpha"), "alpha_fim": _last(g, "alpha"),
            "mu_inicio": _first(g, "mu_total"), "mu_fim": _last(g, "mu_total"), "mu_med": _numeric(g["mu_total"]).median() if "mu_total" in g else np.nan, "mu_max": _numeric(g["mu_total"]).max() if "mu_total" in g else np.nan,
            "dRA_inicio": _first(g, "dRA"), "dRA_fim": _last(g, "dRA"), "dDec_inicio": _first(g, "dDec"), "dDec_fim": _last(g, "dDec"),
        }
        row["Qualidade_noite"] = _quality(row, cfg, int(min_duration_min))
        rows.append(row)
    out = pd.DataFrame(rows)
    audit["objetos_noites"] = int(len(out))
    return out.reset_index(drop=True), audit


def _norm(s: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if not s.notna().any():
        return pd.Series(0.0, index=s.index)
    out = (s - s.min()) / (s.max() - s.min() + 1e-9)
    if invert:
        out = 1.0 - out
    return out.fillna(0).clip(0, 1)


def rank_candidates(summary_night: pd.DataFrame, min_duration_min: int) -> pd.DataFrame:
    if summary_night is None or summary_night.empty:
        return pd.DataFrame()
    df = summary_night[summary_night["Qualidade_noite"].isin(["Boa", "Regular"])].copy()
    df = df[pd.to_numeric(df["Duracao_h"], errors="coerce").fillna(0) * 60 >= float(min_duration_min)]
    if df.empty:
        return pd.DataFrame()
    df["score_brilho"] = _norm(df["V_min"], invert=True)
    df["score_altitude"] = _norm(df["ALT_max"])
    df["score_janela"] = _norm(df["Duracao_h"])
    df["score_vel"] = _norm(df["mu_med"], invert=True)
    df["score_noite"] = 0.35 * df["score_brilho"] + 0.25 * df["score_altitude"] + 0.25 * df["score_janela"] + 0.15 * df["score_vel"]
    best = df.sort_values("score_noite", ascending=False).groupby("Nome_limpo", as_index=False).first()
    n_nights = df.groupby("Nome_limpo")["Noite_UTC"].nunique().rename("n_noites_boas").reset_index()
    best = best.merge(n_nights, on="Nome_limpo", how="left")
    best["score_noites"] = _norm(best["n_noites_boas"])
    best["score_total"] = 0.82 * best["score_noite"] + 0.18 * best["score_noites"]
    best["Melhor_noite"] = best["Noite_UTC"]
    best["Janela_UTC"] = pd.to_datetime(best["Inicio_janela_UTC"]).dt.strftime("%H:%M") + "-" + pd.to_datetime(best["Fim_janela_UTC"]).dt.strftime("%H:%M")
    best["Status_observacional"] = np.where(best["Qualidade_noite"].eq("Boa"), "Bom candidato", "Candidato regular")
    best = best.sort_values("score_total", ascending=False).reset_index(drop=True)
    best.insert(0, "Prioridade", np.arange(1, len(best) + 1))
    return best


def make_color_candidates(ranked_tax: pd.DataFrame, only_without_taxonomy: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    audit = {"entrada": int(len(ranked_tax)) if ranked_tax is not None else 0, "saida": 0, "com_taxonomia": 0, "sem_taxonomia": 0}
    if ranked_tax is None or ranked_tax.empty:
        return pd.DataFrame(), audit
    df = ranked_tax.copy()
    if "Taxonomia disponível" in df.columns and "Taxonomia_encontrada" not in df.columns:
        df["Taxonomia_encontrada"] = df["Taxonomia disponível"]
    if "Classe taxonômica" in df.columns and "Classe_taxonomica" not in df.columns:
        df["Classe_taxonomica"] = df["Classe taxonômica"]
    if "Fonte taxonomia" in df.columns and "Fonte_taxonomia" not in df.columns:
        df["Fonte_taxonomia"] = df["Fonte taxonomia"]
    tax = df.get("Taxonomia_encontrada", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["Status_cores"] = np.where(tax, "Baixa prioridade: já tem taxonomia", "Candidato")
    df["Motivo_status"] = np.where(tax, "ROCKS retornou taxonomia publicada.", "Sem taxonomia publicada encontrada; bom candidato para cores.")
    df["Recomendacao"] = np.where(tax, "Remover da lista principal", "Priorizar")
    audit["com_taxonomia"] = int(tax.sum())
    audit["sem_taxonomia"] = int((~tax).sum())
    if only_without_taxonomy:
        df = df[~tax].copy()
    audit["saida"] = int(len(df))
    return df.sort_values("Prioridade").reset_index(drop=True), audit


def _unk(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return "?"
    except Exception:
        pass
    text = str(value).strip()
    return text if text else "?"


def make_coordinator_support(candidates: pd.DataFrame, project: str = "Eddie") -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    rows = []
    for _, r in candidates.iterrows():
        rows.append({
            "OBJETOS": _unk(r.get("Objeto", r.get("Nome_limpo"))), "D(Km)": _unk(r.get("D_km")), "Prot(h)": _unk(r.get("Prot_h")), "Porb(yr)": _unk(r.get("Porb_yr")), "H": _unk(r.get("H")),
            "Type": "?", "tp": "?", "Spectral": _unk(r.get("Classe_taxonomica")), "Albedo": _unk(r.get("Albedo")),
            "alpha_o": _unk(r.get("alpha_inicio")), "mo": _unk(r.get("V_inicio")), "v_ar_o": _unk(r.get("dRA_inicio")), "v_dec_o": _unk(r.get("dDec_inicio")),
            "alpha_f": _unk(r.get("alpha_fim")), "mf": _unk(r.get("V_fim")), "v_ar_f": _unk(r.get("dRA_fim")), "v_dec_f": _unk(r.get("dDec_fim")),
            "Projects": project, "Tipo_projeto": "C", "Filtros_sugeridos": "g,r,i,z", "Melhor_noite": _unk(r.get("Melhor_noite")), "Janela_UTC": _unk(r.get("Janela_UTC")), "Observacoes": _unk(r.get("Motivo_status")),
        })
    return pd.DataFrame(rows)
