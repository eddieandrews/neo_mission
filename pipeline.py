from __future__ import annotations

import json
import hashlib
import importlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Astropy (já está no requirements do seu projeto)
from astropy import units as u

# Tipos auxiliares
ProgressCB = Optional[Callable[[int, int, str, str], None]]


# =========================
# Configuração (contrato)
# =========================
@dataclass
class ConfigMissao:
    observatorio: str = "Y28"

    # Datas ISO (YYYY-MM-DD)
    data_inicio: str = "2026-01-11"
    data_fim: str = "2026-01-25"

    # Hora UTC opcional para o início (HH:MM). Se None, assume 00:00.
    hora_inicio_utc: Optional[str] = None

    # passo temporal em minutos
    step_min: int = 10

    # Filtros
    ALT_MIN: float = 20.0
    ALT_MAX: float = 70.0
    V_MAX: float = 19.0

    # Filtro opcional de céu escuro (altura do Sol). Se None, não filtra.
    SOL_ALT_MAX: Optional[float] = None

    # Classificação de velocidade
    LIMIAR_RAPIDO: float = 10.0  # arcsec/min

    # Pesos do ranking (somam ~1)
    peso_recencia: float = 0.45
    peso_mag: float = 0.45
    peso_vel: float = 0.10

    # Pastas
    pasta_runs: str = "runs"
    pasta_cache: str = "cache"


def validar_cfg(cfg: ConfigMissao) -> List[str]:
    erros: List[str] = []

    try:
        datetime.fromisoformat(cfg.data_inicio)
    except Exception:
        erros.append("data_inicio inválida (use YYYY-MM-DD).")

    try:
        datetime.fromisoformat(cfg.data_fim)
    except Exception:
        erros.append("data_fim inválida (use YYYY-MM-DD).")

    if cfg.hora_inicio_utc is not None:
        try:
            hh, mm = cfg.hora_inicio_utc.split(":")
            hh = int(hh)
            mm = int(mm)
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError()
        except Exception:
            erros.append("hora_inicio_utc inválida (use HH:MM).")

    if cfg.step_min <= 0:
        erros.append("step_min deve ser > 0.")
    if cfg.ALT_MIN >= cfg.ALT_MAX:
        erros.append("ALT_MIN deve ser menor que ALT_MAX.")
    if cfg.V_MAX <= 0:
        erros.append("V_MAX deve ser > 0.")
    if cfg.LIMIAR_RAPIDO <= 0:
        erros.append("LIMIAR_RAPIDO deve ser > 0.")

    s = float(cfg.peso_recencia + cfg.peso_mag + cfg.peso_vel)
    if not (0.0 < s <= 1.5):
        erros.append("Pesos parecem inválidos. Ajuste peso_recencia/peso_mag/peso_vel.")

    return erros


# =========================
# IO / Run
# =========================
def criar_run_dir(cfg: ConfigMissao) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.pasta_runs) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs").mkdir(exist_ok=True)
    (run_dir / "outputs").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def salvar_manifest(run_dir: Path, cfg: ConfigMissao, inputs: Dict[str, Any], aud: Dict[str, Any]) -> Path:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "cfg": asdict(cfg),
        "inputs": inputs,
        "auditoria": aud,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# =========================
# Etapa 1 — Leitura JPL
# =========================
_JPL_NAME_CANDIDATES = ["Object Name", "Object", "Target", "name", "Name"]


def _normalizar_nome_obj(s: str) -> str:
    s = str(s).strip()
    if not s:
        return s
    s = s.replace("(", "").replace(")", "").strip()
    s = " ".join(s.split())
    return s


def ler_jpl_csvs(paths: List[Path]) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    aud: Dict[str, Any] = {"arquivos": [], "linhas_total": 0, "coluna_usada": None, "objetos_unicos": 0}
    if not paths:
        return pd.DataFrame(), [], {**aud, "erro": "Nenhum arquivo fornecido."}

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
        aud["arquivos"].append(p.name)
        aud["linhas_total"] += int(len(df))

    df_all = pd.concat(dfs, ignore_index=True)

    col = None
    for c in _JPL_NAME_CANDIDATES:
        if c in df_all.columns:
            col = c
            break

    if col is None:
        return df_all, [], {**aud, "erro": f"Nenhuma coluna de nome encontrada. Candidatas: {_JPL_NAME_CANDIDATES}"}

    aud["coluna_usada"] = col
    df_all["Nome do objeto (JPL)"] = df_all[col].astype(str)
    df_all["Nome_limpo"] = df_all["Nome do objeto (JPL)"].map(_normalizar_nome_obj)

    lista_obj = (
        df_all["Nome_limpo"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda x: x != ""]
        .unique()
        .tolist()
    )
    aud["objetos_unicos"] = int(len(lista_obj))
    return df_all, lista_obj, aud


# =========================
# Etapa 2 — MPC via astroquery + cache
# =========================
def _cache_key(cfg: ConfigMissao, obj: str) -> str:
    base = json.dumps(
        {
            "obj": obj,
            "obs": cfg.observatorio,
            "ini": cfg.data_inicio,
            "fim": cfg.data_fim,
            "hora": cfg.hora_inicio_utc,
            "step_min": cfg.step_min,
        },
        sort_keys=True,
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def _parse_start_dt(cfg: ConfigMissao) -> datetime:
    hhmm = cfg.hora_inicio_utc or "00:00"
    t0 = f"{cfg.data_inicio}T{hhmm}:00"
    return datetime.fromisoformat(t0)


def _parse_end_dt(cfg: ConfigMissao) -> datetime:
    hhmm = cfg.hora_inicio_utc or "00:00"
    t1 = f"{cfg.data_fim}T{hhmm}:00"
    return datetime.fromisoformat(t1)


def _compute_number_epochs(cfg: ConfigMissao) -> int:
    dt0 = _parse_start_dt(cfg)
    dt1 = _parse_end_dt(cfg)
    if dt1 < dt0:
        return 0
    step = max(1, int(cfg.step_min))
    total_minutes = int((dt1 - dt0).total_seconds() // 60)
    n = (total_minutes // step) + 1
    return max(1, int(n))


def _mpc_table_to_df(tbl) -> pd.DataFrame:
    try:
        return tbl.to_pandas()
    except Exception:
        try:
            return pd.DataFrame({c: list(tbl[c]) for c in tbl.colnames})
        except Exception:
            return pd.DataFrame()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    colmap_candidates = {
        "V": ["V", "mag", "Mag", "Vmag"],
        "Alt": ["Alt", "alt", "Altitude", "EL", "El", "elev", "Elevation"],
        "SunAlt": ["SunAlt", "sunAlt", "SolAlt", "SunEl", "Sun_EL", "Sun_elev", "SunElevation"],
        "dRA": ["dRA", "RA_rate", "ra_rate", "dRA/dt", "RA motion", "RA_motion"],
        "dDec": ["dDec", "Dec_rate", "dec_rate", "dDec/dt", "Dec motion", "Dec_motion"],
        "mu": ["mu", "pm", "ProperMotion", "proper_motion", "Sky motion", "sky_motion"],
        "epoch": ["epoch", "Epoch", "Date", "date", "datetime", "Time", "time"],
    }

    for std, cands in colmap_candidates.items():
        for c in cands:
            if c in out.columns:
                out.rename(columns={c: std}, inplace=True)
                break

    if "epoch" in out.columns:
        out["epoch"] = pd.to_datetime(out["epoch"], errors="coerce")
    else:
        out["epoch"] = pd.NaT

    for c in ["V", "Alt", "SunAlt", "dRA", "dDec", "mu"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def obter_mpc_astroquery(
    lista_obj: List[str],
    cfg: ConfigMissao,
    run_dir: Path,
    progress_cb: ProgressCB = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    aud: Dict[str, Any] = {
        "total_objetos": int(len(lista_obj)),
        "cache_hits": 0,
        "baixados": 0,
        "falhas": [],
        "mpc_mode": "start_step_number",
        "proper_motion_unit": None,
        "step_quantity": None,
    }

    if not lista_obj:
        return pd.DataFrame(), {**aud, "erro": "lista_obj vazia."}

    cache_dir = Path(cfg.pasta_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from astroquery.mpc import MPC  # type: ignore
    except Exception as e:
        return pd.DataFrame(), {**aud, "erro": f"astroquery.mpc indisponível: {e}"}

    start_dt = _parse_start_dt(cfg)
    number = _compute_number_epochs(cfg)
    location = str(cfg.observatorio)

    # ✅ FIX: step como Quantity (astropy.units)
    step_q = int(cfg.step_min) * u.minute
    aud["step_quantity"] = str(step_q)

    # Vamos tentar proper motion em arcsec/min; se não aceitar, caímos em arcsec/h e convertemos.
    proper_motion_unit_tried = ["arcsec/min", "arcsec/h"]

    total = max(1, len(lista_obj))
    all_rows: List[pd.DataFrame] = []

    for i, obj in enumerate(lista_obj, start=1):
        if progress_cb:
            progress_cb(i, total, obj, "cache/check")

        key = _cache_key(cfg, obj)
        safe_name = obj.replace("/", "_").replace(" ", "_")
        p_cache = cache_dir / f"mpc_{safe_name}_{key}.parquet"

        if p_cache.exists():
            try:
                df_obj = pd.read_parquet(p_cache)
                aud["cache_hits"] += 1
                all_rows.append(df_obj)
                continue
            except Exception:
                pass

        if progress_cb:
            progress_cb(i, total, obj, "download")

        last_err = None
        df_obj = None
        used_unit = None

        for pm_unit in proper_motion_unit_tried:
            try:
                tbl = MPC.get_ephemeris(
                    target=obj,
                    location=location,
                    start=start_dt.isoformat(sep=" "),
                    step=step_q,          # ✅ Quantity
                    number=number,
                    proper_motion="total",
                    proper_motion_unit=pm_unit,
                    cache=False,
                )
                used_unit = pm_unit
                df_obj = _standardize_columns(_mpc_table_to_df(tbl))
                break
            except Exception as e:
                last_err = e
                df_obj = None

        if df_obj is None or df_obj.empty:
            aud["falhas"].append({"object": obj, "erro": str(last_err) if last_err else "Falha desconhecida"})
            continue

        df_obj["Nome_limpo"] = obj

        # se veio mu em arcsec/h, converte para arcsec/min (padrão interno)
        if "mu" in df_obj.columns and used_unit == "arcsec/h":
            df_obj["mu"] = df_obj["mu"] / 60.0

        aud["proper_motion_unit"] = used_unit
        aud["baixados"] += 1

        try:
            df_obj.to_parquet(p_cache, index=False)
        except Exception:
            pass

        all_rows.append(df_obj)

    if not all_rows:
        return pd.DataFrame(), aud

    df_all = pd.concat(all_rows, ignore_index=True)

    if "epoch" not in df_all.columns:
        df_all["epoch"] = pd.NaT
    else:
        df_all["epoch"] = pd.to_datetime(df_all["epoch"], errors="coerce")

    return df_all, aud


# =========================
# Etapa 3 — Filtros / Classe / Resumo / Ranking
# =========================
def filtrar_epocas(df_mpc: pd.DataFrame, cfg: ConfigMissao) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    aud: Dict[str, Any] = {
        "linhas_entrada": int(len(df_mpc)) if df_mpc is not None else 0,
        "filtro_V_aplicado": False,
        "filtro_ALT_aplicado": False,
        "filtro_Sol_aplicado": False,
        "linhas_saida": 0,
        "avisos": [],
    }

    if df_mpc is None or df_mpc.empty:
        return pd.DataFrame(), {**aud, "avisos": ["df_mpc vazio."]}

    df = df_mpc.copy()

    if "V" in df.columns:
        aud["filtro_V_aplicado"] = True
        df = df[df["V"].notna() & (df["V"] <= float(cfg.V_MAX))]
    else:
        aud["avisos"].append("Coluna 'V' não encontrada; filtro de magnitude não aplicado.")

    if "Alt" in df.columns:
        aud["filtro_ALT_aplicado"] = True
        df = df[df["Alt"].notna() & (df["Alt"] >= float(cfg.ALT_MIN)) & (df["Alt"] <= float(cfg.ALT_MAX))]
    else:
        aud["avisos"].append("Coluna 'Alt' não encontrada; filtros de altitude não aplicados.")

    if cfg.SOL_ALT_MAX is not None:
        if "SunAlt" in df.columns:
            aud["filtro_Sol_aplicado"] = True
            df = df[df["SunAlt"].notna() & (df["SunAlt"] <= float(cfg.SOL_ALT_MAX))]
        else:
            aud["avisos"].append("SOL_ALT_MAX definido, mas coluna 'SunAlt' não existe; filtro do Sol não aplicado.")

    aud["linhas_saida"] = int(len(df))
    return df.reset_index(drop=True), aud


def classificar_velocidade(df_obs: pd.DataFrame, cfg: ConfigMissao) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
    aud: Dict[str, Any] = {
        "linhas_entrada": int(len(df_obs)) if df_obs is not None else 0,
        "modo_velocidade": None,
        "limiar_rapido": float(cfg.LIMIAR_RAPIDO),
        "linhas_com_vel": 0,
        "avisos": [],
    }
    classe_obj: Dict[str, str] = {}

    if df_obs is None or df_obs.empty:
        return pd.DataFrame(), classe_obj, {**aud, "avisos": ["df_obs vazio."]}

    df = df_obs.copy()

    if "mu" in df.columns and df["mu"].notna().any():
        df["mu_total"] = df["mu"]
        aud["modo_velocidade"] = "mu_total"
    elif ("dRA" in df.columns and "dDec" in df.columns) and (df["dRA"].notna().any() or df["dDec"].notna().any()):
        df["mu_total"] = np.sqrt(df["dRA"].fillna(0.0) ** 2 + df["dDec"].fillna(0.0) ** 2)
        aud["modo_velocidade"] = "componentes(dRA,dDec)"
    else:
        df["mu_total"] = np.nan
        aud["modo_velocidade"] = "indisponivel"
        aud["avisos"].append("Sem colunas de velocidade (mu ou dRA/dDec). Classificação rápido/lento não aplicada.")

    aud["linhas_com_vel"] = int(df["mu_total"].notna().sum())

    lim = float(cfg.LIMIAR_RAPIDO)
    df["Classe_vel"] = np.where(
        df["mu_total"].notna() & (df["mu_total"] > lim),
        "rapido",
        np.where(df["mu_total"].notna(), "lento", "desconhecida"),
    )

    if "Nome_limpo" in df.columns:
        for obj, g in df.groupby("Nome_limpo"):
            counts = g["Classe_vel"].value_counts(dropna=False)
            classe_obj[str(obj)] = str(counts.idxmax()) if len(counts) else "desconhecida"

    return df.reset_index(drop=True), classe_obj, aud


def resumir_por_objeto(df_obs: pd.DataFrame, cfg: ConfigMissao) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    aud: Dict[str, Any] = {"objetos": 0, "avisos": []}
    if df_obs is None or df_obs.empty:
        return pd.DataFrame(), {**aud, "avisos": ["df_obs vazio."]}

    if "Nome_limpo" not in df_obs.columns:
        return pd.DataFrame(), {**aud, "avisos": ["Coluna Nome_limpo ausente."]}

    df = df_obs.copy()
    agg: Dict[str, Tuple[str, str]] = {}

    if "V" in df.columns:
        agg["V_min"] = ("V", "min")
        agg["V_med"] = ("V", "median")
    else:
        df["V"] = np.nan
        agg["V_min"] = ("V", "min")
        agg["V_med"] = ("V", "median")
        aud["avisos"].append("Sem coluna V; V_min/V_med serão NaN.")

    if "Alt" in df.columns:
        agg["ALT_max"] = ("Alt", "max")
        agg["ALT_med"] = ("Alt", "median")
    else:
        df["Alt"] = np.nan
        agg["ALT_max"] = ("Alt", "max")
        agg["ALT_med"] = ("Alt", "median")
        aud["avisos"].append("Sem coluna Alt; ALT_max/ALT_med serão NaN.")

    if "mu_total" in df.columns:
        agg["mu_med"] = ("mu_total", "median")
        agg["mu_max"] = ("mu_total", "max")
    else:
        df["mu_total"] = np.nan
        agg["mu_med"] = ("mu_total", "median")
        agg["mu_max"] = ("mu_total", "max")
        aud["avisos"].append("Sem mu_total; mu_med/mu_max serão NaN.")

    agg["n_epocas"] = ("Nome_limpo", "size")

    if "epoch" in df.columns:
        agg["epoch_min"] = ("epoch", "min")
        agg["epoch_max"] = ("epoch", "max")
    else:
        df["epoch"] = pd.NaT
        agg["epoch_min"] = ("epoch", "min")
        agg["epoch_max"] = ("epoch", "max")

    out = df.groupby("Nome_limpo", dropna=False).agg(**agg).reset_index()
    out["Nome do objeto"] = out["Nome_limpo"]

    aud["objetos"] = int(len(out))
    return out, aud


def ranquear(summary: pd.DataFrame, cfg: ConfigMissao) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()

    df = summary.copy()

    v = pd.to_numeric(df["V_min"], errors="coerce")
    if v.notna().any():
        v_norm = (v.max() - v) / (v.max() - v.min() + 1e-9)
        v_norm = v_norm.fillna(0.0)
    else:
        v_norm = pd.Series(0.0, index=df.index)
    df["score_mag"] = v_norm

    mu = pd.to_numeric(df["mu_med"], errors="coerce")
    if mu.notna().any():
        mu_norm = (mu.max() - mu) / (mu.max() - mu.min() + 1e-9)
        mu_norm = mu_norm.fillna(0.0)
    else:
        mu_norm = pd.Series(0.0, index=df.index)
    df["score_vel"] = mu_norm

    t = pd.to_datetime(df["epoch_max"], errors="coerce")
    if t.notna().any():
        t_ord = t.map(lambda x: x.timestamp() if pd.notna(x) else np.nan).astype(float)
        t_norm = (t_ord - np.nanmin(t_ord)) / (np.nanmax(t_ord) - np.nanmin(t_ord) + 1e-9)
        t_norm = pd.Series(t_norm, index=df.index).fillna(0.0)
    else:
        t_norm = pd.Series(0.0, index=df.index)
    df["score_recencia"] = t_norm

    df["score_total"] = (
        float(cfg.peso_recencia) * df["score_recencia"]
        + float(cfg.peso_mag) * df["score_mag"]
        + float(cfg.peso_vel) * df["score_vel"]
    )

    df = df.sort_values("score_total", ascending=False).reset_index(drop=True)
    df.insert(0, "Prioridade", np.arange(1, len(df) + 1))
    return df


# =========================
# Taxonomia (ROCKS) — enriquecimento opcional pós-Etapa 3
# =========================
def _pick_first_nonempty(data: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = data.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _extract_taxonomy_info(payload: Any) -> Tuple[bool, Optional[str], Optional[str], Any]:
    if payload is None:
        return False, None, None, None

    if isinstance(payload, list):
        if len(payload) == 0:
            return False, None, None, []
        first = payload[0]
        if isinstance(first, dict):
            classe = _pick_first_nonempty(first, ["class", "name", "value", "label", "type"])
            fonte = _pick_first_nonempty(first, ["source", "reference", "ref", "bibcode"])
            return classe is not None, classe, fonte, payload
        return True, str(first), None, payload

    if isinstance(payload, dict):
        classe = _pick_first_nonempty(payload, ["class", "name", "value", "label", "type", "taxonomy"])
        fonte = _pick_first_nonempty(payload, ["source", "reference", "ref", "bibcode"])
        return classe is not None, classe, fonte, payload

    text = str(payload).strip()
    return bool(text), text if text else None, None, text


def _query_taxonomy_with_rocks(object_name: str) -> Dict[str, Any]:
    try:
        rocks_mod = importlib.import_module("rocks")
    except Exception:
        return {
            "status": "rocks_unavailable",
            "has_taxonomy": False,
            "taxonomy_class": None,
            "taxonomy_source": None,
            "taxonomy_raw": None,
            "error": "Pacote 'rocks' não está disponível no ambiente.",
        }

    try:
        rock_obj = rocks_mod.Rock(object_name)
        tax_payload = getattr(rock_obj, "taxonomy", None)
        has_tax, tax_class, tax_source, tax_raw = _extract_taxonomy_info(tax_payload)
        return {
            "status": "ok",
            "has_taxonomy": has_tax,
            "taxonomy_class": tax_class,
            "taxonomy_source": tax_source,
            "taxonomy_raw": tax_raw,
            "error": None,
        }
    except Exception as e:
        return {
            "status": "query_error",
            "has_taxonomy": False,
            "taxonomy_class": None,
            "taxonomy_source": None,
            "taxonomy_raw": None,
            "error": str(e),
        }


def enriquecer_taxonomia_rocks(
    ranked: pd.DataFrame,
    progress_cb: ProgressCB = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    aud: Dict[str, Any] = {
        "objetos_entrada": 0,
        "objetos_consultados": 0,
        "objetos_com_taxonomia": 0,
        "objetos_sem_taxonomia": 0,
        "falhas": [],
        "rocks_disponivel": None,
    }

    if ranked is None or ranked.empty:
        return pd.DataFrame(), aud

    if "Nome_limpo" not in ranked.columns:
        raise KeyError("Tabela ranqueada sem coluna 'Nome_limpo'.")

    df = ranked.copy()
    objs = df["Nome_limpo"].astype(str).str.strip().dropna().unique().tolist()
    aud["objetos_entrada"] = int(len(objs))

    cache: Dict[str, Dict[str, Any]] = {}
    total = max(1, len(objs))

    for i, obj in enumerate(objs, start=1):
        if progress_cb:
            progress_cb(i, total, obj, "taxonomia")

        res = _query_taxonomy_with_rocks(obj)
        cache[obj] = res
        aud["objetos_consultados"] += 1

        if res.get("status") != "ok":
            aud["falhas"].append({"object": obj, "erro": res.get("error"), "status": res.get("status")})

    df["Taxonomia disponível"] = df["Nome_limpo"].map(
        lambda x: bool(cache.get(str(x).strip(), {}).get("has_taxonomy", False))
    )
    df["Classe taxonômica"] = df["Nome_limpo"].map(
        lambda x: cache.get(str(x).strip(), {}).get("taxonomy_class")
    )
    df["Fonte taxonomia"] = df["Nome_limpo"].map(
        lambda x: cache.get(str(x).strip(), {}).get("taxonomy_source")
    )

    aud["objetos_com_taxonomia"] = int(df[df["Taxonomia disponível"]]["Nome_limpo"].nunique())
    aud["objetos_sem_taxonomia"] = int(df[~df["Taxonomia disponível"]]["Nome_limpo"].nunique())
    aud["rocks_disponivel"] = len([f for f in aud["falhas"] if f.get("status") == "rocks_unavailable"]) == 0
    return df, aud
