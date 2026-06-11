from __future__ import annotations

import importlib
import shutil
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


def rocks_diagnostic(sample_objects: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"installed": False, "import_error": None, "tests": []}
    try:
        rocks = importlib.import_module("rocks")
        result["installed"] = True
        result["module"] = str(getattr(rocks, "__file__", "?"))
        result["version"] = str(getattr(rocks, "__version__", "?"))
    except Exception as e:
        result["import_error"] = str(e)
        return result

    for raw in sample_objects[:8]:
        info = parse_object_identifier(raw)
        row = {
            "entrada": raw,
            "identificador_preferido": info.get("identificador_preferido"),
            "tipo": info.get("tipo_identificador_preferido"),
            "tentativas": identifier_variants(raw, include_name=True),
            "ok": False,
            "target_usado": None,
            "erro": None,
            "tem_taxonomia": None,
        }
        last_err = None
        for target in row["tentativas"]:
            try:
                obj = rocks.Rock(target)
                tax = getattr(obj, "taxonomy", None)
                row["ok"] = True
                row["target_usado"] = target
                row["tem_taxonomia"] = tax is not None and str(tax).strip() not in ["", "[]", "None"]
                break
            except Exception as e:
                last_err = e
        if not row["ok"]:
            row["erro"] = str(last_err) if last_err else "Falha desconhecida"
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
