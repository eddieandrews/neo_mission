from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from astropy import units as u

from pipeline import ConfigMissao, _compute_number_epochs, _mpc_table_to_df, _parse_start_dt, _standardize_columns

ProgressCB = Optional[Callable[[int, int, str, str], None]]

_PROV_RE = re.compile(r"\b((?:19|20)\d{2}\s+[A-Z]{1,3}\d{0,3}[A-Z]?)\b")
_PACKED_RE = re.compile(r"\b([A-Z]\d{3}\s+[A-Z]{1,3})\b")


def mpc_query_variants(object_name: str) -> List[str]:
    """Gera identificadores alternativos para o MPC.

    Muitos CSVs trazem nomes como '1036 Ganymed A924 UB' ou
    '398188 Agni 2010 LE15'. O MPC geralmente aceita melhor o numero
    ('1036') ou a designacao ('2010 LE15') do que a string completa.
    """
    raw = str(object_name).replace("(", "").replace(")", "").replace("*", "").strip()
    raw = " ".join(raw.split())
    if not raw:
        return []

    out: List[str] = []
    tokens = raw.split()

    # Numero MPC no inicio. Evita interpretar designacoes como '2021 VR3' como numero.
    if tokens and tokens[0].isdigit():
        n = int(tokens[0])
        if not (1900 <= n <= 2099 and len(tokens) == 2):
            out.append(tokens[0])

    # Designacao provisoria classica, ex.: 2010 LE15, 2004 FE31, 2021 VR3.
    for match in _PROV_RE.findall(raw):
        out.append(match.strip())

    # Designacao packed/cometa antiga que aparece em alguns exportes, ex.: A924 UB.
    for match in _PACKED_RE.findall(raw):
        out.append(match.strip())

    out.append(raw)

    clean: List[str] = []
    seen = set()
    for item in out:
        item = " ".join(str(item).split())
        key = item.lower()
        if item and key not in seen:
            clean.append(item)
            seen.add(key)
    return clean


def _cache_key(cfg: ConfigMissao, obj: str) -> str:
    base = json.dumps(
        {
            "obj": obj,
            "obs": cfg.observatorio,
            "ini": cfg.data_inicio,
            "fim": cfg.data_fim,
            "hora": cfg.hora_inicio_utc,
            "step_min": cfg.step_min,
            "resolver": "variants_v1",
        },
        sort_keys=True,
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def obter_mpc_astroquery_resiliente(
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
        "identificadores_usados": [],
        "mpc_mode": "start_step_number_variants",
        "proper_motion_unit": None,
        "step_quantity": None,
    }

    if not lista_obj:
        return pd.DataFrame(), {**aud, "erro": "lista_obj vazia."}

    try:
        from astroquery.mpc import MPC  # type: ignore
    except Exception as e:
        return pd.DataFrame(), {**aud, "erro": f"astroquery.mpc indisponivel: {e}"}

    cache_dir = Path(cfg.pasta_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_dt = _parse_start_dt(cfg)
    number = _compute_number_epochs(cfg)
    location = str(cfg.observatorio)
    step_q = int(cfg.step_min) * u.minute
    aud["step_quantity"] = str(step_q)

    proper_motion_unit_tried = ["arcsec/min", "arcsec/h"]
    total = max(1, len(lista_obj))
    all_rows: List[pd.DataFrame] = []

    for i, obj in enumerate(lista_obj, start=1):
        variants = mpc_query_variants(obj)
        if progress_cb:
            progress_cb(i, total, obj, "cache/check")

        key = _cache_key(cfg, obj)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(obj)).strip("_")[:80]
        p_cache = cache_dir / f"mpc_{safe_name}_{key}.parquet"

        if p_cache.exists():
            try:
                df_obj = pd.read_parquet(p_cache)
                aud["cache_hits"] += 1
                all_rows.append(df_obj)
                continue
            except Exception:
                pass

        last_err = None
        df_obj = None
        used_unit = None
        used_target = None

        for target in variants:
            if progress_cb:
                progress_cb(i, total, obj, f"download: {target}")
            for pm_unit in proper_motion_unit_tried:
                try:
                    tbl = MPC.get_ephemeris(
                        target=target,
                        location=location,
                        start=start_dt.isoformat(sep=" "),
                        step=step_q,
                        number=number,
                        proper_motion="total",
                        proper_motion_unit=pm_unit,
                        cache=False,
                    )
                    df_obj = _standardize_columns(_mpc_table_to_df(tbl))
                    used_unit = pm_unit
                    used_target = target
                    break
                except Exception as e:
                    last_err = e
                    df_obj = None
            if df_obj is not None and not df_obj.empty:
                break

        if df_obj is None or df_obj.empty:
            aud["falhas"].append(
                {
                    "object": obj,
                    "tentativas": variants,
                    "erro": str(last_err) if last_err else "Falha desconhecida",
                }
            )
            continue

        df_obj["Nome_limpo"] = obj
        df_obj["MPC_target_usado"] = used_target

        if "mu" in df_obj.columns and used_unit == "arcsec/h":
            df_obj["mu"] = df_obj["mu"] / 60.0

        aud["proper_motion_unit"] = used_unit
        aud["baixados"] += 1
        aud["identificadores_usados"].append({"object": obj, "target_usado": used_target, "tentativas": variants})

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
