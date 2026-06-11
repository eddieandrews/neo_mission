from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from object_ids import identifier_variants, parse_object_identifier

ProgressCB = Optional[Callable[[int, int, str, str], None]]


def _pick_first_nonempty(data: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _extract_taxonomy(payload: Any) -> Tuple[bool, Optional[str], Optional[str], Any]:
    if payload is None:
        return False, None, None, None
    if isinstance(payload, list):
        if len(payload) == 0:
            return False, None, None, []
        payload0 = payload[0]
        if isinstance(payload0, dict):
            cls = _pick_first_nonempty(payload0, ["class", "name", "value", "label", "type", "taxonomy"])
            src = _pick_first_nonempty(payload0, ["source", "reference", "ref", "bibcode", "shortbib"])
            return cls is not None, cls, src, payload
        text = str(payload0).strip()
        return bool(text), text or None, None, payload
    if isinstance(payload, dict):
        cls = _pick_first_nonempty(payload, ["class", "name", "value", "label", "type", "taxonomy"])
        src = _pick_first_nonempty(payload, ["source", "reference", "ref", "bibcode", "shortbib"])
        return cls is not None, cls, src, payload
    text = str(payload).strip()
    return bool(text), text or None, None, text


def _safe_attr(obj: Any, names: List[str]) -> Any:
    for name in names:
        try:
            value = getattr(obj, name)
            if value is not None and str(value).strip() != "":
                return value
        except Exception:
            pass
    return None


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        for key in ["value", "val", "mean", "preferred", "diameter", "albedo", "H", "period"]:
            if key in value:
                return _clean(value[key])
        return str(value)
    if isinstance(value, list):
        return _clean(value[0]) if value else None
    return str(value)


def _query_one(rocks_mod: Any, target: str) -> Dict[str, Any]:
    rock_obj = rocks_mod.Rock(target)
    tax_payload = getattr(rock_obj, "taxonomy", None)
    has_tax, tax_class, tax_source, tax_raw = _extract_taxonomy(tax_payload)
    return {
        "status": "ok",
        "has_taxonomy": has_tax,
        "taxonomy_class": tax_class,
        "taxonomy_source": tax_source,
        "taxonomy_raw": tax_raw,
        "D_km": _clean(_safe_attr(rock_obj, ["diameter", "diameter_km", "D"])),
        "Albedo": _clean(_safe_attr(rock_obj, ["albedo", "geometric_albedo", "pv"])),
        "H": _clean(_safe_attr(rock_obj, ["H", "absolute_magnitude", "absolute_magnitude_H"])),
        "Prot_h": _clean(_safe_attr(rock_obj, ["rotation_period", "rotational_period", "period", "period_rotation"])),
        "Porb_yr": _clean(_safe_attr(rock_obj, ["orbital_period", "period_orbit"])),
        "error": None,
    }


def query_rocks_resilient(object_name: str) -> Dict[str, Any]:
    info = parse_object_identifier(object_name)
    variants = identifier_variants(object_name, include_name=True)

    try:
        rocks_mod = importlib.import_module("rocks")
    except Exception:
        return {
            "status": "rocks_unavailable",
            "has_taxonomy": False,
            "taxonomy_class": None,
            "taxonomy_source": None,
            "taxonomy_raw": None,
            "rocks_target_usado": None,
            "identificador_preferido": info.get("identificador_preferido"),
            "numero_oficial": info.get("numero_oficial"),
            "designacao_provisoria": info.get("designacao_provisoria"),
            "tentativas": variants,
            "error": "Pacote 'rocks' não está disponível no ambiente.",
        }

    last_err = None
    for target in variants:
        try:
            result = _query_one(rocks_mod, target)
            result.update(
                {
                    "rocks_target_usado": target,
                    "identificador_preferido": info.get("identificador_preferido"),
                    "numero_oficial": info.get("numero_oficial"),
                    "designacao_provisoria": info.get("designacao_provisoria"),
                    "tentativas": variants,
                }
            )
            return result
        except Exception as e:
            last_err = e

    return {
        "status": "query_error",
        "has_taxonomy": False,
        "taxonomy_class": None,
        "taxonomy_source": None,
        "taxonomy_raw": None,
        "rocks_target_usado": None,
        "identificador_preferido": info.get("identificador_preferido"),
        "numero_oficial": info.get("numero_oficial"),
        "designacao_provisoria": info.get("designacao_provisoria"),
        "tentativas": variants,
        "error": str(last_err) if last_err else "Falha desconhecida",
    }


def enriquecer_taxonomia_rocks_resiliente(
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
        info = parse_object_identifier(obj)
        if progress_cb:
            progress_cb(i, total, str(info.get("identificador_preferido") or obj), "taxonomia")
        res = query_rocks_resilient(obj)
        cache[obj] = res
        aud["objetos_consultados"] += 1
        if res.get("status") != "ok":
            aud["falhas"].append({"object": obj, "identificador_preferido": res.get("identificador_preferido"), "tentativas": res.get("tentativas"), "erro": res.get("error"), "status": res.get("status")})

    df["Identificador_preferido"] = df["Nome_limpo"].map(lambda x: cache.get(str(x).strip(), {}).get("identificador_preferido"))
    df["Numero_oficial"] = df["Nome_limpo"].map(lambda x: cache.get(str(x).strip(), {}).get("numero_oficial"))
    df["Designacao_provisoria"] = df["Nome_limpo"].map(lambda x: cache.get(str(x).strip(), {}).get("designacao_provisoria"))
    df["ROCKS_target_usado"] = df["Nome_limpo"].map(lambda x: cache.get(str(x).strip(), {}).get("rocks_target_usado"))
    df["Taxonomia disponível"] = df["Nome_limpo"].map(lambda x: bool(cache.get(str(x).strip(), {}).get("has_taxonomy", False)))
    df["Classe taxonômica"] = df["Nome_limpo"].map(lambda x: cache.get(str(x).strip(), {}).get("taxonomy_class"))
    df["Fonte taxonomia"] = df["Nome_limpo"].map(lambda x: cache.get(str(x).strip(), {}).get("taxonomy_source"))
    df["Taxonomia_status_consulta"] = df["Nome_limpo"].map(lambda x: cache.get(str(x).strip(), {}).get("status"))
    df["Taxonomia_erro"] = df["Nome_limpo"].map(lambda x: cache.get(str(x).strip(), {}).get("error"))

    for col in ["D_km", "Albedo", "H", "Prot_h", "Porb_yr"]:
        df[col] = df["Nome_limpo"].map(lambda x, c=col: cache.get(str(x).strip(), {}).get(c))

    aud["objetos_com_taxonomia"] = int(df[df["Taxonomia disponível"]]["Nome_limpo"].nunique())
    aud["objetos_sem_taxonomia"] = int(df[~df["Taxonomia disponível"]]["Nome_limpo"].nunique())
    aud["rocks_disponivel"] = len([f for f in aud["falhas"] if f.get("status") == "rocks_unavailable"]) == 0
    return df, aud
