from __future__ import annotations

import importlib
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from object_ids import identifier_variants, parse_object_identifier

ProgressCB = Optional[Callable[[int, int, str, str], None]]


def _force_threaded_resolver() -> str:
    """Evita falhas do aiohttp/aiodns no Windows quando o DNS do sistema funciona.

    O space-rocks usa chamadas HTTP assíncronas. Em alguns ambientes Windows,
    o resolver async baseado em aiodns/pycares falha com 'Could not contact DNS servers',
    mesmo quando nslookup e Test-NetConnection funcionam. Forçamos o aiohttp a usar
    o resolver em thread, baseado no socket/getaddrinfo do sistema operacional.
    """
    try:
        import aiohttp.resolver as aio_resolver  # type: ignore
        if hasattr(aio_resolver, "ThreadedResolver"):
            aio_resolver.DefaultResolver = aio_resolver.ThreadedResolver
            return "aiohttp.ThreadedResolver"
        return "aiohttp.resolver_sem_ThreadedResolver"
    except Exception as exc:
        return f"resolver_patch_failed: {exc}"


def _system_dns_test(host: str = "ssp.imcce.fr") -> str:
    try:
        return str(socket.getaddrinfo(host, 443)[0][4][0])
    except Exception as exc:
        return f"socket_dns_failed: {exc}"


def _val(x: Any) -> Any:
    if x is None:
        return None
    if hasattr(x, "value"):
        try:
            return _val(x.value)
        except Exception:
            pass
    if isinstance(x, pd.Series):
        y = x.dropna().tolist()
        return _val(y[0]) if y else None
    if isinstance(x, (list, tuple)):
        return _val(x[0]) if x else None
    return x


def _attr(obj: Any, path: List[str]) -> Any:
    cur = obj
    for name in path:
        try:
            cur = getattr(cur, name)
        except Exception:
            return None
    out = _val(cur)
    if out is None or str(out).strip() == "":
        return None
    return out


def _first(obj: Any, paths: List[List[str]]) -> Any:
    for path in paths:
        out = _attr(obj, path)
        if out is not None:
            return out
    return None


def _table(table: Any, cols: List[str]) -> Any:
    try:
        df = pd.DataFrame(table)
    except Exception:
        return None
    if df.empty:
        return None
    for col in cols:
        if col in df.columns:
            vals = df[col].dropna().tolist()
            if vals:
                return vals[0]
    return None


def _import_rocks() -> Tuple[Any, Optional[str], str, str]:
    resolver_patch = _force_threaded_resolver()
    dns_ip = _system_dns_test()
    try:
        rocks_mod = importlib.import_module("rocks")
    except Exception as exc:
        return None, f"Nao importou rocks/space-rocks: {exc}", resolver_patch, dns_ip
    if not hasattr(rocks_mod, "Rock") or not hasattr(rocks_mod, "id"):
        return None, "O modulo rocks importado nao possui Rock/id; instale space-rocks.", resolver_patch, dns_ip
    return rocks_mod, None, resolver_patch, dns_ip


def _resolve(rocks_mod: Any, variants: List[str]) -> Tuple[Any, Optional[str], Optional[str]]:
    for target in variants:
        try:
            name, number = rocks_mod.id(target)
            if number is not None and str(number).lower() not in ["nan", "none", ""]:
                return number, str(name), target
            if name is not None and str(name).strip():
                return target, str(name), target
        except Exception:
            pass
    return None, None, None


def query_rocks_resilient(object_name: str) -> Dict[str, Any]:
    info = parse_object_identifier(object_name)
    variants = identifier_variants(object_name, include_name=True)
    base = {
        "identificador_preferido": info.get("identificador_preferido"),
        "tipo_identificador_preferido": info.get("tipo_identificador_preferido"),
        "numero_oficial": info.get("numero_oficial"),
        "designacao_provisoria": info.get("designacao_provisoria"),
        "tentativas": variants,
    }
    rocks_mod, err, resolver_patch, dns_ip = _import_rocks()
    base.update({"dns_resolver_usado": resolver_patch, "python_dns_ssp": dns_ip})
    if err:
        return {**base, "status": "space_rocks_unavailable", "has_taxonomy": False, "error": err}

    resolved_id, resolved_name, resolved_from = _resolve(rocks_mod, variants)
    target = resolved_id or info.get("identificador_preferido") or variants[0]
    try:
        rock = rocks_mod.Rock(target, datacloud="taxonomies")
        tax_table = getattr(rock, "taxonomies", None)
        tax_class = _first(rock, [["taxonomy", "class_"], ["parameters", "physical", "taxonomy", "class_"]]) or _table(tax_table, ["class_", "class", "taxonomy"])
        tax_rows = len(pd.DataFrame(tax_table)) if tax_table is not None else 0
        return {
            **base,
            "status": "ok",
            "has_taxonomy": tax_class is not None and str(tax_class).strip() != "",
            "taxonomy_class": tax_class,
            "taxonomy_complex": _first(rock, [["taxonomy", "complex"]]) or _table(tax_table, ["complex"]),
            "taxonomy_scheme": _first(rock, [["taxonomy", "scheme"]]) or _table(tax_table, ["scheme"]),
            "taxonomy_method": _table(tax_table, ["method"]),
            "taxonomy_waverange": _table(tax_table, ["waverange", "wavelength"]),
            "taxonomy_source": _first(rock, [["taxonomy", "shortbib"], ["taxonomy", "bibcode"]]) or _table(tax_table, ["shortbib", "bibcode", "reference"]),
            "taxonomy_datacloud_rows": int(tax_rows),
            "rocks_target_usado": target,
            "rocks_resolved_name": resolved_name,
            "rocks_resolved_number": resolved_id,
            "rocks_resolved_from": resolved_from,
            "D_km": _first(rock, [["diameter"], ["parameters", "physical", "diameter"]]),
            "Albedo": _first(rock, [["albedo"], ["parameters", "physical", "albedo"]]),
            "H": _first(rock, [["absolute_magnitude"], ["H"]]),
            "Prot_h": _first(rock, [["rotation_period"], ["rotational_period"], ["period"]]),
            "Porb_yr": _first(rock, [["orbital_period"], ["P"]]),
            "error": None,
        }
    except Exception as exc:
        return {**base, "status": "query_error", "has_taxonomy": False, "rocks_target_usado": target, "rocks_resolved_name": resolved_name, "rocks_resolved_number": resolved_id, "rocks_resolved_from": resolved_from, "error": str(exc)}


def enriquecer_taxonomia_rocks_resiliente(ranked: pd.DataFrame, progress_cb: ProgressCB = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    aud: Dict[str, Any] = {"objetos_entrada": 0, "objetos_consultados": 0, "objetos_com_taxonomia": 0, "objetos_sem_taxonomia": 0, "falhas": [], "rocks_disponivel": None, "metodo": "space-rocks Rock(datacloud=taxonomies) + aiohttp ThreadedResolver"}
    if ranked is None or ranked.empty:
        return pd.DataFrame(), aud
    if "Nome_limpo" not in ranked.columns:
        raise KeyError("Tabela ranqueada sem coluna Nome_limpo.")
    df = ranked.copy()
    objs = df["Nome_limpo"].astype(str).str.strip().dropna().unique().tolist()
    aud["objetos_entrada"] = int(len(objs))
    cache: Dict[str, Dict[str, Any]] = {}
    for i, obj in enumerate(objs, start=1):
        if progress_cb:
            progress_cb(i, len(objs), obj, "space-rocks")
        res = query_rocks_resilient(obj)
        cache[obj] = res
        aud["objetos_consultados"] += 1
        if res.get("status") != "ok":
            aud["falhas"].append({"object": obj, "status": res.get("status"), "erro": res.get("error"), "tentativas": res.get("tentativas"), "dns_resolver_usado": res.get("dns_resolver_usado"), "python_dns_ssp": res.get("python_dns_ssp")})
    mapping = {
        "Identificador_preferido": "identificador_preferido", "Tipo_identificador_preferido": "tipo_identificador_preferido", "Numero_oficial": "numero_oficial", "Designacao_provisoria": "designacao_provisoria",
        "ROCKS_target_usado": "rocks_target_usado", "ROCKS_resolved_name": "rocks_resolved_name", "ROCKS_resolved_number": "rocks_resolved_number", "ROCKS_resolved_from": "rocks_resolved_from", "DNS_resolver_usado": "dns_resolver_usado", "Python_DNS_ssp": "python_dns_ssp",
        "Taxonomia disponível": "has_taxonomy", "Classe taxonômica": "taxonomy_class", "Complexo taxonômico": "taxonomy_complex", "Esquema taxonômico": "taxonomy_scheme", "Metodo taxonomia": "taxonomy_method", "Faixa taxonomia": "taxonomy_waverange", "Fonte taxonomia": "taxonomy_source", "Taxonomia_datacloud_rows": "taxonomy_datacloud_rows", "Taxonomia_status_consulta": "status", "Taxonomia_erro": "error",
        "D_km": "D_km", "Albedo": "Albedo", "H": "H", "Prot_h": "Prot_h", "Porb_yr": "Porb_yr",
    }
    for out_col, key in mapping.items():
        df[out_col] = df["Nome_limpo"].map(lambda x, k=key: cache.get(str(x).strip(), {}).get(k))
    df["Taxonomia disponível"] = df["Taxonomia disponível"].fillna(False).astype(bool)
    aud["objetos_com_taxonomia"] = int(df[df["Taxonomia disponível"]]["Nome_limpo"].nunique())
    aud["objetos_sem_taxonomia"] = int(df[~df["Taxonomia disponível"]]["Nome_limpo"].nunique())
    aud["rocks_disponivel"] = len([f for f in aud["falhas"] if f.get("status") == "space_rocks_unavailable"]) == 0
    return df, aud
