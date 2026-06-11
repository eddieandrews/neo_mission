from __future__ import annotations

import importlib
import importlib.util
import shutil
import socket
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from object_ids import parse_object_identifier, identifier_variants


def cache_stats(cache_dir: str = "cache") -> Dict[str, Any]:
    path = Path(cache_dir)
    files = list(path.glob("*.parquet")) if path.exists() else []
    total_bytes = sum(f.stat().st_size for f in files if f.exists())
    return {
        "cache_dir": str(path),
        "exists": path.exists(),
        "parquet_files": len(files),
        "total_mb": round(total_bytes / (1024 * 1024), 3),
    }


def clear_cache(cache_dir: str = "cache") -> Dict[str, Any]:
    path = Path(cache_dir)
    before = cache_stats(cache_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    after = cache_stats(cache_dir)
    return {"before": before, "after": after}


def package_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def environment_diagnostic() -> Dict[str, Any]:
    """Diagnostica o ambiente que afeta o space-rocks.

    No Windows, a presença de aiodns/pycares pode fazer o aiohttp usar um
    resolvedor DNS diferente do resolver do sistema. Se isso ocorrer, o
    space-rocks pode falhar com 'Could not contact DNS servers' mesmo quando
    nslookup e Test-NetConnection funcionam.
    """
    out: Dict[str, Any] = {
        "aiodns_instalado": package_present("aiodns"),
        "pycares_instalado": package_present("pycares"),
        "recomendacao": None,
        "python_dns_ssp": None,
        "rocks_importado": False,
        "rocks_module": None,
        "rocks_version": None,
    }
    try:
        out["python_dns_ssp"] = socket.getaddrinfo("ssp.imcce.fr", 443)[0][4][0]
    except Exception as exc:
        out["python_dns_ssp"] = f"falhou: {exc}"

    try:
        rocks = importlib.import_module("rocks")
        out["rocks_importado"] = True
        out["rocks_module"] = str(getattr(rocks, "__file__", "?"))
        out["rocks_version"] = str(getattr(rocks, "__version__", "?"))
        out["rocks_tem_Rock"] = hasattr(rocks, "Rock")
        out["rocks_tem_id"] = hasattr(rocks, "id")
    except Exception as exc:
        out["rocks_import_error"] = str(exc)

    if out["aiodns_instalado"] or out["pycares_instalado"]:
        out["recomendacao"] = "Remover aiodns/pycares neste ambiente: pip uninstall aiodns pycares -y"
    else:
        out["recomendacao"] = "OK: aiodns/pycares ausentes; space-rocks deve usar o DNS padrao do sistema."
    return out


def rocks_diagnostic(sample_objects: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = environment_diagnostic()
    result["tests"] = []
    if not result.get("rocks_importado"):
        return result

    # Usa a mesma função do pipeline real para o diagnóstico ser fiel ao app.
    from rocks_fix import query_rocks_resilient

    for raw in sample_objects[:8]:
        info = parse_object_identifier(raw)
        row = {
            "entrada": raw,
            "identificador_preferido": info.get("identificador_preferido"),
            "tipo": info.get("tipo_identificador_preferido"),
            "tentativas": identifier_variants(raw, include_name=True),
        }
        try:
            res = query_rocks_resilient(raw)
            row.update(
                {
                    "status": res.get("status"),
                    "ok": res.get("status") == "ok",
                    "target_usado": res.get("rocks_target_usado"),
                    "tem_taxonomia": res.get("has_taxonomy"),
                    "classe": res.get("taxonomy_class"),
                    "datacloud_rows": res.get("taxonomy_datacloud_rows"),
                    "erro": res.get("error"),
                    "dns_resolver_usado": res.get("dns_resolver_usado"),
                    "python_dns_ssp": res.get("python_dns_ssp"),
                }
            )
        except Exception as exc:
            row.update({"status": "diagnostic_error", "ok": False, "erro": str(exc)})
        result["tests"].append(row)
    return result


def identifiers_audit_df(values: List[str]) -> pd.DataFrame:
    rows = []
    for value in values:
        info = parse_object_identifier(value)
        rows.append(
            {
                "entrada": value,
                "identificador_preferido": info.get("identificador_preferido"),
                "tipo": info.get("tipo_identificador_preferido"),
                "numero_oficial": info.get("numero_oficial"),
                "designacao_provisoria": info.get("designacao_provisoria"),
                "designacao_packed": info.get("designacao_packed"),
                "nome_proprio": info.get("nome_proprio"),
                "tentativas": ", ".join(identifier_variants(value, include_name=True)),
            }
        )
    return pd.DataFrame(rows)
