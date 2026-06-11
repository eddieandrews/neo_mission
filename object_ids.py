from __future__ import annotations

import re
from typing import Dict, List, Optional

# Designação provisória descompactada: 1949 MA, 1998 EC3, 2010 LE15, 2021 VR3, 2000 GQ146.
_PROV_RE = re.compile(r"\b((?:19|20)\d{2}\s+[A-Z]{1,3}\d{0,4}[A-Z]?)\b")

# Designação packed/antiga vista em alguns catálogos: A924 UB.
_PACKED_SPACED_RE = re.compile(r"\b([A-Z]\d{3}\s+[A-Z]{1,3})\b")

# Designação packed sem espaço, comum em MPC/JPL para provisórios modernos: K25A00A etc.
_PACKED_COMPACT_RE = re.compile(r"\b([IJK]\d{2}[A-Z]\d{2}[A-Z])\b")

# Número oficial entre parênteses ou colchetes: (1566), [1566].
_PAREN_NUMBER_RE = re.compile(r"[\(\[]\s*(\d{1,7})\s*[\)\]]")


def normalize_object_name(value: str) -> str:
    text = str(value).replace("*", " ").strip()
    text = text.replace(";", " ").replace(",", " ")
    return " ".join(text.split())


def _strip_wrappers(value: str) -> str:
    return str(value).replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")


def _is_year_token(token: str) -> bool:
    if not token.isdigit():
        return False
    n = int(token)
    return 1800 <= n <= 2099


def _extract_official_number(raw: str, tokens: List[str]) -> Optional[str]:
    # 1. Preferência máxima: número explícito em parênteses/colchetes.
    m = _PAREN_NUMBER_RE.search(raw)
    if m:
        return m.group(1)

    # 2. Número como primeiro campo, exceto quando for uma designação provisória tipo '2021 VR3'.
    if tokens and tokens[0].isdigit():
        if not (_is_year_token(tokens[0]) and len(tokens) >= 2 and re.match(r"^[A-Z]{1,3}\d{0,4}[A-Z]?$", tokens[1])):
            return tokens[0]

    # 3. Número em outro ponto do nome, útil para 'Icarus 1566 1949 MA'.
    #    Ignora anos de designação provisória.
    for i, token in enumerate(tokens):
        if not token.isdigit():
            continue
        if _is_year_token(token):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if re.match(r"^[A-Z]{1,3}\d{0,4}[A-Z]?$", nxt):
                continue
        return token

    return None


def parse_object_identifier(value: str) -> Dict[str, Optional[str]]:
    """Extrai identificadores úteis de uma string de objeto.

    Regra operacional:
    1. número oficial MPC, quando existir;
    2. designação provisória descompactada, quando não houver número;
    3. designação packed;
    4. nome original normalizado.

    Exemplos:
    - '1566 Icarus 1949 MA' -> 1566
    - '(1566) Icarus' -> 1566
    - 'Icarus 1566 1949 MA' -> 1566
    - '2021 VR3' -> 2021 VR3
    - '1036 Ganymed A924 UB' -> 1036
    """
    raw = normalize_object_name(value)
    clean_no_paren = normalize_object_name(_strip_wrappers(raw))
    tokens = clean_no_paren.split()

    numero_oficial = _extract_official_number(raw, tokens)

    prov = None
    prov_matches = _PROV_RE.findall(clean_no_paren)
    if prov_matches:
        prov = prov_matches[-1].strip()

    packed = None
    packed_matches = _PACKED_SPACED_RE.findall(clean_no_paren)
    if packed_matches:
        packed = packed_matches[-1].strip()
    else:
        compact_matches = _PACKED_COMPACT_RE.findall(clean_no_paren)
        if compact_matches:
            packed = compact_matches[-1].strip()

    nome_proprio = clean_no_paren
    for part in [numero_oficial, prov, packed]:
        if part:
            nome_proprio = nome_proprio.replace(part, " ")
    nome_proprio = " ".join(nome_proprio.split()) or None

    preferido = numero_oficial or prov or packed or clean_no_paren or None
    tipo = "numero_oficial" if numero_oficial else "designacao_provisoria" if prov else "designacao_packed" if packed else "nome_original"

    return {
        "nome_original": clean_no_paren,
        "numero_oficial": numero_oficial,
        "designacao_provisoria": prov,
        "designacao_packed": packed,
        "nome_proprio": nome_proprio,
        "identificador_preferido": preferido,
        "tipo_identificador_preferido": tipo,
    }


def identifier_variants(value: str, include_name: bool = True) -> List[str]:
    info = parse_object_identifier(value)
    candidates = [
        info.get("numero_oficial"),
        info.get("designacao_provisoria"),
        info.get("designacao_packed"),
    ]
    if include_name:
        candidates.extend([info.get("nome_proprio"), info.get("nome_original")])
    else:
        candidates.append(info.get("nome_original"))

    out: List[str] = []
    seen = set()
    for item in candidates:
        if item is None:
            continue
        text = normalize_object_name(item)
        text = normalize_object_name(_strip_wrappers(text))
        key = text.lower()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def identifiers_dataframe(values: List[str]):
    """Tabela de auditoria para mostrar no Streamlit."""
    import pandas as pd

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
