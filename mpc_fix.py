from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from astropy import units as u

from object_ids import identifier_variants, parse_object_identifier
from pipeline import ConfigMissao, _compute_number_epochs, _mpc_table_to_df, _parse_start_dt, _standardize_columns

ProgressCB = Optional[Callable[[int, int, str, str], None]]


def mpc_query_variants(object_name: str) -> List[str]:
    """Identificadores para consulta MPC, em ordem de preferência.

    Regra:
    1. número oficial MPC, se existir;
    2. designação provisória, se existir;
    3. designação packed, se existir;
    4. nome próprio/original como última alternativa.
    """
    return identifier_variants(object_name, include_name=True)


def _cache_key(cfg: ConfigMissao, obj: str) -> str:
    preferred = parse_object_identifier(obj).get("identificador_preferido") or str(obj)
    base = json.dumps(
        {
            "obj": preferred,
            "obj_original": str(obj),
            "obs": cfg.observatorio,
            "ini": cfg.data_inicio,
            "fim": cfg.data_fim,
            "hora": cfg.hora_inicio_utc,
            "step_min": cfg.step_min,
            "resolver": "preferred_official_v2",
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
        "mpc_mode": "start_step_number_preferred_official",
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
        id_info = parse_object_identifier(obj)
        variants = mpc_query_variants(obj)
        if progress_cb:
            progress_cb(i, total, obj, "cache/check")

        key = _cache_key(cfg, obj)
        safe_preferred = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(id_info.get("identificador_preferido") or obj)).strip("_")[:80]
        p_cache = cache_dir / f"mpc_{safe_preferred}_{key}.parquet"

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
                    "identificador_preferido": id_info.get("identificador_preferido"),
                    "numero_oficial": id_info.get("numero_oficial"),
                    "designacao_provisoria": id_info.get("designacao_provisoria"),
                    "tentativas": variants,
                    "erro": str(last_err) if last_err else "Falha desconhecida",
                }
            )
            continue

        # Mantém o nome original para rastreabilidade, mas adiciona o identificador limpo para MPC/ROCKS.
        df_obj["Nome_limpo"] = obj
        df_obj["Objeto_original"] = id_info.get("nome_original") or obj
        df_obj["Identificador_preferido"] = id_info.get("identificador_preferido") or used_target
        df_obj["Numero_oficial"] = id_info.get("numero_oficial")
        df_obj["Designacao_provisoria"] = id_info.get("designacao_provisoria")
        df_obj["MPC_target_usado"] = used_target

        if "mu" in df_obj.columns and used_unit == "arcsec/h":
            df_obj["mu"] = df_obj["mu"] / 60.0

        aud["proper_motion_unit"] = used_unit
        aud["baixados"] += 1
        aud["identificadores_usados"].append(
            {
                "object": obj,
                "identificador_preferido": id_info.get("identificador_preferido"),
                "target_usado": used_target,
                "tentativas": variants,
            }
        )

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
