from __future__ import annotations

import re
from typing import Dict, List, Optional

_PROV_RE = re.compile(r"\b((?:19|20)\d{2}\s+[A-Z]{1,3}\d{0,3}[A-Z]?)\b")
_PACKED_RE = re.compile(r"\b([A-Z]\d{3}\s+[A-Z]{1,3})\b")


def normalize_object_name(value: str) -> str:
    text = str(value).replace("(", "").replace(")", "").replace("*", " ").strip()
    return " ".join(text.split())


def parse_object_identifier(value: str) -> Dict[str, Optional[str]]:
    """Extrai identificadores úteis de uma linha de objeto.

    Exemplos:
    - '1566 Icarus 1949 MA' -> numero_oficial='1566', designacao_provisoria='1949 MA'
    - '2021 VR3' -> numero_oficial=None, designacao_provisoria='2021 VR3'
    - '1036 Ganymed A924 UB' -> numero_oficial='1036', designacao_packed='A924 UB'

    A preferência operacional é sempre:
    1. número oficial MPC;
    2. designação provisória;
    3. designação packed;
    4. nome original normalizado.
    """
    raw = normalize_object_name(value)
    tokens = raw.split()

    numero_oficial: Optional[str] = None
    if tokens and tokens[0].isdigit():
        n = int(tokens[0])
        # '2021 VR3' é designação provisória, não número oficial.
        if not (1900 <= n <= 2099 and len(tokens) == 2):
            numero_oficial = tokens[0]

    prov = None
    prov_matches = _PROV_RE.findall(raw)
    if prov_matches:
        prov = prov_matches[-1].strip()

    packed = None
    packed_matches = _PACKED_RE.findall(raw)
    if packed_matches:
        packed = packed_matches[-1].strip()

    nome_proprio = raw
    if numero_oficial and nome_proprio.startswith(numero_oficial):
        nome_proprio = nome_proprio[len(numero_oficial):].strip()
    if prov:
        nome_proprio = nome_proprio.replace(prov, "").strip()
    if packed:
        nome_proprio = nome_proprio.replace(packed, "").strip()
    nome_proprio = " ".join(nome_proprio.split()) or None

    preferido = numero_oficial or prov or packed or raw or None

    return {
        "nome_original": raw,
        "numero_oficial": numero_oficial,
        "designacao_provisoria": prov,
        "designacao_packed": packed,
        "nome_proprio": nome_proprio,
        "identificador_preferido": preferido,
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
        key = text.lower()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out
