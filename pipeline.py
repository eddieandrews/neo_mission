from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd


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
    # Ex.: "18:00" -> começa em 2026-01-11T18:00:00Z.
    hora_inicio_utc: Optional[str] = None

    # passo temporal em minutos
    step_min: int = 10

    ALT_MIN: float = 20.0
    ALT_MAX: float = 70.0
    V_MAX: float = 19.0

    # (opcional) filtrar apenas céu escuro via altura do Sol (graus)
    # Ex.: -18 (astronômico), -12 (náutico), -6 (civil)
    # Se None, não filtra por Sol.
    SOL_ALT_MAX: Optional[float] = None

    # classificação rápido/lento usando mu_total ("/min)
    LIMIAR_RAPIDO: float = 10.0

    # ranking
    peso_recencia: float = 0.45
    peso_mag: float = 0.45
    peso_vel: float = 0.10

    ANALISE_PADRAO: str = "CORES (g', r', i', z')"

    pasta_runs: str = "runs"
    pasta_cache: str = "cache"


# =========================
# Utilidades / auditoria
# =========================
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validar_cfg(cfg: ConfigMissao) -> List[str]:
    erros: List[str] = []
    try:
        di = date.fromisoformat(cfg.data_inicio)
        df = date.fromisoformat(cfg.data_fim)
        if di > df:
            erros.append("data_inicio > data_fim (inverta as datas).")
    except Exception:
        erros.append("Datas inválidas. Use formato ISO: YYYY-MM-DD.")

    if cfg.hora_inicio_utc is not None and str(cfg.hora_inicio_utc).strip() != "":
        try:
            hh, mm = str(cfg.hora_inicio_utc).strip().split(":")
            hh = int(hh)
            mm = int(mm)
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError
        except Exception:
            erros.append("hora_inicio_utc inválida. Use HH:MM (ex.: 18:00) ou deixe vazio.")

    if cfg.step_min not in (5, 10, 15, 20, 30, 60):
        erros.append("step_min inválido. Use 5, 10, 15, 20, 30 ou 60.")

    if not (0 <= cfg.ALT_MIN < cfg.ALT_MAX <= 90):
        erros.append("ALT_MIN/ALT_MAX inválidos (esperado 0 ≤ ALT_MIN < ALT_MAX ≤ 90).")

    if not (10 <= cfg.V_MAX <= 22):
        erros.append("V_MAX fora do razoável (esperado entre 10 e 22).")

    if cfg.LIMIAR_RAPIDO <= 0:
        erros.append("LIMIAR_RAPIDO deve ser > 0.")

    if cfg.SOL_ALT_MAX is not None:
        try:
            sol = float(cfg.SOL_ALT_MAX)
            if not (-90 <= sol <= 90):
                erros.append("SOL_ALT_MAX fora do intervalo [-90, 90].")
        except Exception:
            erros.append("SOL_ALT_MAX deve ser numérico ou vazio.")

    s = cfg.peso_recencia + cfg.peso_mag + cfg.peso_vel
    if abs(s - 1.0) > 1e-6:
        erros.append(f"Pesos do ranking devem somar 1.0 (atual = {s}).")

    return erros


# =========================
# JPL: leitura + nomes
# =========================
def ler_jpl_csvs(paths: List[Path]) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    aud: Dict[str, Any] = {}
    if not paths:
        raise FileNotFoundError("Nenhum CSV do JPL foi fornecido.")

    rows: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")
        df = pd.read_csv(p)
        rows.append(df)

    df_all = pd.concat(rows, ignore_index=True)

    candidatos = ["Object Name", "Object", "Target", "name", "Name"]
    col_name = None
    for c in candidatos:
        if c in df_all.columns:
            col_name = c
            break
    if col_name is None:
        raise KeyError(f"Não encontrei coluna de nome no JPL. Colunas disponíveis: {list(df_all.columns)}")

    nomes_raw = df_all[col_name].astype(str).tolist()
    nomes_norm = [normalizar_nome_para_mpc(x) for x in nomes_raw]
    nomes_norm = [x for x in nomes_norm if x]
    lista_unica = sorted(pd.unique(nomes_norm).tolist())

    aud["jpl_total_linhas"] = int(len(df_all))
    aud["jpl_coluna_nome"] = col_name
    aud["jpl_objetos_unicos"] = int(len(lista_unica))
    aud["jpl_amostra_objetos"] = lista_unica[:15]

    return df_all, lista_unica, aud


def normalizar_nome_para_mpc(s: str) -> str:
    """
    Normalizações típicas:
    - "(2025 XC2)" -> "2025 XC2"
    - "21088 Chelyabinsk (1992 BL2)" -> "1992 BL2"
    - "2025 WA3" mantém
    """
    if s is None:
        return ""
    t = str(s).strip()

    m = None
    if "(" in t and ")" in t:
        m = list(pd.Series([t]).str.extract(r".*\(([^()]*)\)\s*$").iloc[0])[0]
    if m and isinstance(m, str) and m.strip():
        t = m.strip()

    t = t.replace("(", "").replace(")", "").strip()
    t = " ".join(t.split())
    return t


# =========================
# MPC: aquisição (astroquery) — versão robusta + callback
# =========================
ProgressCB = Optional[Callable[[int, int, str, str], None]]
# assinatura: (i_atual, total, objeto, fase) -> None


def _montar_start_time(cfg: ConfigMissao) -> str:
    """
    Monta start como string/time do astropy, com hora UTC opcional.
    """
    if cfg.hora_inicio_utc and str(cfg.hora_inicio_utc).strip():
        hhmm = str(cfg.hora_inicio_utc).strip()
        return f"{cfg.data_inicio} {hhmm}"
    return cfg.data_inicio


def obter_mpc_astroquery(
    lista_objetos: List[str],
    cfg: ConfigMissao,
    run_dir: Path,  # mantido por compatibilidade com o app (não usado aqui)
    progress_cb: ProgressCB = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Baixa efemérides via astroquery.mpc com cache por objeto.
    Retorna dataframe padronizado com colunas:
      object, dt_utc, V, alt, mu_total ("/min), ano_desc
    (Opcional) se o MPC retornar, também traz:
      sun_alt
    """
    from astroquery.mpc import MPC
    from astropy.time import Time
    import astropy.units as u
    import time

    aud: Dict[str, Any] = {
        "mpc_modo": "astroquery",
        "objetos_entrada": int(len(lista_objetos)),
        "objetos_cache": 0,
        "objetos_baixados": 0,
        "objetos_falha": 0,
        "falhas": [],
    }

    cache_dir = Path(cfg.pasta_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # step robusto como Quantity
    try:
        step_min = float(cfg.step_min)
    except Exception:
        raise ValueError(f"cfg.step_min inválido: {cfg.step_min!r}. Deve ser numérico (minutos).")
    if not np.isfinite(step_min) or step_min <= 0:
        raise ValueError(f"cfg.step_min inválido: {cfg.step_min!r}. Deve ser > 0.")

    step_q = float(step_min) * u.min

    # start como astropy.time.Time (mais estável)
    start_str = _montar_start_time(cfg)
    start_t = Time(start_str)

    aud["step_min_normalizado"] = step_min
    aud["step_q"] = str(step_q)
    aud["start_t_isot"] = str(start_t.isot)

    # janela temporal e passos
    di = datetime.fromisoformat(cfg.data_inicio)
    df = datetime.fromisoformat(cfg.data_fim)
    total_min = int((df - di).total_seconds() // 60)
    n_steps = int(max(1, total_min // int(round(step_min)) + 1))
    aud["n_steps"] = int(n_steps)

    frames: List[pd.DataFrame] = []
    t0_all = time.time()

    total = int(len(lista_objetos))

    # Lista de candidatos de colunas (MPC pode variar a nomenclatura)
    time_candidates = ["Date", "Date (UT)", "UT", "Datetime", "date"]
    v_candidates = ["V", "Mag", "mag", "Vmag"]
    alt_candidates = ["Altitude", "Alt", "alt", "EL", "Elev"]
    pm_candidates = [
        "Proper motion",
        "Proper Motion",
        "PM",
        "Motion",
        "mu",
        "Sky motion",
        "Sky Motion",
    ]
    sun_candidates = ["Sun altitude", "Sun Altitude", "Sun alt", "SunAlt", "Sun_Altitude"]

    for i, obj in enumerate(lista_objetos, start=1):
        obj_clean = str(obj).strip()
        if not obj_clean:
            continue

        if progress_cb:
            progress_cb(i, total, obj_clean, "início")

        cache_file = cache_dir / (
            f"mpc_{obj_clean.replace(' ', '_')}_{cfg.data_inicio}_{cfg.data_fim}_{int(round(step_min))}min.parquet"
        )

        try:
            if cache_file.exists():
                df_obj = pd.read_parquet(cache_file)
                aud["objetos_cache"] += 1
                if progress_cb:
                    progress_cb(i, total, obj_clean, "cache")
            else:
                last_err = None
                df_obj = None

                for attempt in range(1, 4):
                    try:
                        if progress_cb:
                            progress_cb(i, total, obj_clean, f"baixando (tentativa {attempt}/3)")

                        eph = MPC.get_ephemeris(
                            obj_clean,
                            location=str(cfg.observatorio).strip(),
                            start=start_t,
                            step=step_q,
                            number=int(n_steps),
                        )
                        df_obj = eph.to_pandas()
                        aud["objetos_baixados"] += 1
                        break
                    except Exception as e:
                        last_err = str(e)
                        time.sleep(1.5 * attempt)

                if df_obj is None:
                    raise RuntimeError(last_err or "Falha desconhecida ao consultar MPC.")

                df_obj.to_parquet(cache_file, index=False)

            if i == 1:
                aud["colunas_retorno_mpc_exemplo"] = list(df_obj.columns)

            # 1) tempo
            time_col = next((c for c in time_candidates if c in df_obj.columns), None)
            if time_col is None:
                raise KeyError(f"{obj_clean}: não encontrei coluna de data/hora. Colunas: {list(df_obj.columns)}")

            dt = pd.to_datetime(df_obj[time_col], errors="coerce", utc=True)
            if dt.isna().all():
                raise ValueError(
                    f"{obj_clean}: datas não parsearam (coluna {time_col}). Amostra: {df_obj[time_col].head(3).tolist()}"
                )

            # 2) magnitude V
            v_col = next((c for c in v_candidates if c in df_obj.columns), None)
            if v_col is None:
                raise KeyError(f"{obj_clean}: não encontrei coluna de magnitude V. Colunas: {list(df_obj.columns)}")
            V = pd.to_numeric(df_obj[v_col], errors="coerce")

            # 3) altitude (objeto)
            alt_col = next((c for c in alt_candidates if c in df_obj.columns), None)
            if alt_col is None:
                raise KeyError(f"{obj_clean}: não encontrei coluna de altitude. Colunas: {list(df_obj.columns)}")
            alt = pd.to_numeric(df_obj[alt_col], errors="coerce")

            # 4) proper motion (módulo)
            pm_col = next((c for c in pm_candidates if c in df_obj.columns), None)
            if pm_col is None:
                raise KeyError(f"{obj_clean}: não encontrei coluna de proper motion. Colunas: {list(df_obj.columns)}")
            pm = pd.to_numeric(df_obj[pm_col], errors="coerce")

            # unidade heurística "/h" vs "/min"
            med = float(pm.dropna().median()) if pm.notna().any() else np.nan
            if np.isfinite(med) and med > 50:
                mu_total = pm / 60.0
                unidade_assumida = '"/h → "/min'
            else:
                mu_total = pm.copy()
                unidade_assumida = 'assumido "/min'

            # (opcional) altura do Sol
            sun_col = next((c for c in sun_candidates if c in df_obj.columns), None)
            sun_alt = pd.to_numeric(df_obj[sun_col], errors="coerce") if sun_col else None

            # ano de designação
            ano_desc = pd.to_numeric(
                pd.Series([obj_clean]).str.extract(r"(\d{4})")[0],
                errors="coerce"
            ).iloc[0]

            out_dict: Dict[str, Any] = {
                "object": obj_clean,
                "dt_utc": dt.dt.tz_convert(None),  # datetime naive em UTC
                "V": V,
                "alt": alt,
                "mu_total": mu_total,
                "ano_desc": ano_desc,
            }
            if sun_alt is not None:
                out_dict["sun_alt"] = sun_alt

            out = pd.DataFrame(out_dict).dropna(subset=["dt_utc"])

            if i == 1:
                aud["pm_unidade_heuristica"] = unidade_assumida
                if sun_col is not None:
                    aud["sun_alt_col"] = sun_col

            frames.append(out)

            if progress_cb:
                progress_cb(i, total, obj_clean, "ok")

        except Exception as e:
            aud["objetos_falha"] += 1
            aud["falhas"].append({"object": obj_clean, "erro": str(e)})
            if progress_cb:
                progress_cb(i, total, obj_clean, "falha")

    if not frames:
        cols = ["object", "dt_utc", "V", "alt", "mu_total", "ano_desc"]
        if cfg.SOL_ALT_MAX is not None:
            cols.append("sun_alt")
        return pd.DataFrame(columns=cols), aud

    df_all = pd.concat(frames, ignore_index=True)

    # garante dtype de dt_utc
    df_all["dt_utc"] = pd.to_datetime(df_all["dt_utc"], errors="coerce")
    df_all = df_all.dropna(subset=["dt_utc"]).reset_index(drop=True)

    aud["linhas_mpc_total"] = int(len(df_all))
    aud["objetos_com_dados"] = int(df_all["object"].nunique())
    aud["tempo_total_s"] = round(time.time() - t0_all, 2)

    aud["V_min"] = float(df_all["V"].min()) if df_all["V"].notna().any() else None
    aud["V_max"] = float(df_all["V"].max()) if df_all["V"].notna().any() else None
    aud["ALT_min"] = float(df_all["alt"].min()) if df_all["alt"].notna().any() else None
    aud["ALT_max"] = float(df_all["alt"].max()) if df_all["alt"].notna().any() else None
    aud["mu_med"] = float(df_all["mu_total"].median()) if df_all["mu_total"].notna().any() else None
    aud["mu_max"] = float(df_all["mu_total"].max()) if df_all["mu_total"].notna().any() else None

    if "sun_alt" in df_all.columns:
        aud["SUN_ALT_min"] = float(df_all["sun_alt"].min()) if df_all["sun_alt"].notna().any() else None
        aud["SUN_ALT_max"] = float(df_all["sun_alt"].max()) if df_all["sun_alt"].notna().any() else None
    elif cfg.SOL_ALT_MAX is not None:
        # você pediu filtro de Sol, mas o MPC não trouxe a coluna
        aud["aviso_sol_alt"] = "SOL_ALT_MAX foi configurado, mas o MPC não retornou 'Sun altitude'. Filtro solar não aplicado."

    return df_all, aud


# =========================
# Filtros + resumo + ranking
# =========================
def filtrar_epocas(df: pd.DataFrame, cfg: ConfigMissao) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    aud: Dict[str, Any] = {}
    if df.empty:
        aud["linhas_total"] = 0
        return df, aud

    aud["linhas_total"] = int(len(df))

    mask_v = df["V"] <= cfg.V_MAX
    mask_alt = (df["alt"] >= cfg.ALT_MIN) & (df["alt"] <= cfg.ALT_MAX)

    # filtro opcional por Sol
    if cfg.SOL_ALT_MAX is not None:
        if "sun_alt" in df.columns:
            sol_lim = float(cfg.SOL_ALT_MAX)
            mask_sun = df["sun_alt"] <= sol_lim
            aud["linhas_SOL_ok"] = int(mask_sun.sum())
        else:
            # não filtra, mas audita
            mask_sun = pd.Series(True, index=df.index)
            aud["linhas_SOL_ok"] = None
            aud["aviso_sol_alt"] = "SOL_ALT_MAX configurado, mas df não tem coluna 'sun_alt'. Filtro solar não aplicado."
    else:
        mask_sun = pd.Series(True, index=df.index)
        aud["linhas_SOL_ok"] = None

    aud["linhas_V_ok"] = int(mask_v.sum())
    aud["linhas_ALT_ok"] = int(mask_alt.sum())
    aud["linhas_VeALT"] = int((mask_v & mask_alt).sum())

    df_obs = (
        df[mask_v & mask_alt & mask_sun]
        .copy()
        .sort_values(["object", "dt_utc"])
        .reset_index(drop=True)
    )
    aud["linhas_VeALTeSOL"] = int(len(df_obs))
    aud["objetos_obs"] = int(df_obs["object"].nunique()) if not df_obs.empty else 0
    return df_obs, aud


def classificar_velocidade(df_obs: pd.DataFrame, cfg: ConfigMissao) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    aud: Dict[str, Any] = {}
    if df_obs.empty:
        return df_obs, pd.DataFrame(columns=["object", "classe_objeto"]), {"objetos": 0}

    df_obs = df_obs.copy()
    df_obs["classe_linha"] = np.where(df_obs["mu_total"] >= cfg.LIMIAR_RAPIDO, "rápido", "lento")

    classe_obj = (
        df_obs.assign(eh_rapido=df_obs["mu_total"] >= cfg.LIMIAR_RAPIDO)
        .groupby("object", as_index=False)["eh_rapido"]
        .max()
    )
    classe_obj["classe_objeto"] = np.where(classe_obj["eh_rapido"], "rápido", "lento")
    classe_obj = classe_obj[["object", "classe_objeto"]]

    df_obs = df_obs.merge(classe_obj, on="object", how="left")

    aud["objetos_total"] = int(df_obs["object"].nunique())
    aud["rapidos_obj"] = int((classe_obj["classe_objeto"] == "rápido").sum())
    aud["lentos_obj"] = int((classe_obj["classe_objeto"] == "lento").sum())
    return df_obs, classe_obj, aud


def resumir_por_objeto(df_obs: pd.DataFrame, cfg: ConfigMissao) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    aud: Dict[str, Any] = {}
    if df_obs.empty:
        return pd.DataFrame(), {"objetos": 0}

    df_obs = df_obs.copy()
    df_obs["date_utc"] = df_obs["dt_utc"].dt.date

    rows: List[Dict[str, Any]] = []
    for obj, g in df_obs.groupby("object", sort=True):
        g = g.sort_values("dt_utc").reset_index(drop=True)
        ini = g.iloc[0]
        ultimo_dia = g["date_utc"].max()
        primeiro_dia = g["date_utc"].min()
        fim = g[g["date_utc"] == ultimo_dia].iloc[0]
        best = g.loc[g["V"].idxmin()]

        janela = f"{primeiro_dia.day:02d}–{ultimo_dia.day:02d}/{ultimo_dia.month:02d}"
        deltaV = f"{float(ini['V']):.1f}–{float(fim['V']):.1f}"

        nome_saida = obj + ("*" if ini["classe_objeto"] == "rápido" else "")

        rows.append({
            "Nome do objeto": nome_saida,
            "Nome_limpo": obj,
            "Melhores dias para observação": janela,
            "ΔV-MAG (Início–Fim)": deltaV,
            "Análise": cfg.ANALISE_PADRAO,
            'μ ("/min) Início': float(ini["mu_total"]),
            'μ ("/min) Fim': float(fim["mu_total"]),
            "Classe": ini["classe_objeto"],
            "Melhor dia (menor V)": best["dt_utc"].strftime("%Y-%m-%d"),
            "V no melhor ponto": float(best["V"]),
            "ALT no melhor ponto (deg)": float(best["alt"]),
            "Ano": float(ini.get("ano_desc", np.nan)),
        })

    summary = pd.DataFrame(rows)
    aud["objetos_resumo"] = int(len(summary))
    return summary, aud


def ranquear(summary: pd.DataFrame, cfg: ConfigMissao) -> pd.DataFrame:
    df = summary.copy()

    if "Ano" not in df.columns:
        df["Ano"] = np.nan

    if df["Ano"].isna().all():
        df["Ano"] = df["Nome_limpo"].str.extract(r"(\d{4})").astype(float)

    ano = df["Ano"]
    if ano.notna().sum() >= 2:
        min_y, max_y = float(ano.min()), float(ano.max())
        df["score_recencia"] = (ano - min_y) / (max_y - min_y) if max_y > min_y else 0.0
    else:
        df["score_recencia"] = 0.0

    v = df["V no melhor ponto"]
    vmin, vmax = float(v.min()), float(v.max())
    df["score_mag"] = 1 - (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0

    df["score_vel"] = df["Classe"].map({"rápido": 1.0, "lento": 0.0}).fillna(0.0)

    df["score_total"] = (
        cfg.peso_recencia * df["score_recencia"] +
        cfg.peso_mag * df["score_mag"] +
        cfg.peso_vel * df["score_vel"]
    )

    df = df.sort_values("score_total", ascending=False).reset_index(drop=True)
    df.insert(0, "Prioridade", np.arange(1, len(df) + 1))
    return df


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
