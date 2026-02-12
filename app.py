import streamlit as st
from pathlib import Path
import pandas as pd
import time

from pipeline import (
    ConfigMissao, validar_cfg, criar_run_dir, salvar_manifest,
    ler_jpl_csvs, filtrar_epocas, classificar_velocidade,
    resumir_por_objeto, ranquear, obter_mpc_astroquery
)

st.set_page_config(page_title="Y28 NEO Mission Pipeline", layout="wide")
st.title("Y28 — Seleção de NEOs para Cores (Pipeline Auditável)")


# =========================
# Helpers: status persistente por etapa
# =========================
def _init_status():
    if "status" not in st.session_state:
        st.session_state.status = {
            "run": None,
            "etapa1": {"ok": False, "msg": "", "aud": None},
            "etapa2": {"ok": False, "msg": "", "aud": None},
            "etapa3": {"ok": False, "msg": "", "aud": None},
            "etapa4": {"ok": False, "msg": "", "aud": None},
        }


def _render_status_cards():
    st.subheader("Status do pipeline (persistente)")
    cols = st.columns(4)
    etapa_map = [
        ("Etapa 1", "etapa1"),
        ("Etapa 2", "etapa2"),
        ("Etapa 3", "etapa3"),
        ("Etapa 4", "etapa4"),
    ]
    for c, (label, key) in zip(cols, etapa_map):
        with c:
            box = st.container(border=True)
            ok = st.session_state.status[key]["ok"]
            msg = st.session_state.status[key]["msg"] or "—"
            box.markdown(f"**{label}**")
            if ok:
                box.success(msg)
            else:
                box.info(msg)


def _salvar_manifest_atual(run_dir: Path, cfg: ConfigMissao, jpl_paths):
    """
    Salva o manifesto sempre com a auditoria consolidada mais recente.
    Mantém compatibilidade mesmo se não existir etapa de Taxonomia no app atual.
    """
    aud_etapa3 = (st.session_state.status.get("etapa3", {}) or {}).get("aud", {}) or {}
    aud_total = {
        "JPL": st.session_state.get("aud_jpl", {}) or {},
        "MPC": st.session_state.get("aud_mpc", {}) or {},
        "Filtros": aud_etapa3.get("Filtros"),
        "Classe": aud_etapa3.get("Classe"),
        "Resumo": aud_etapa3.get("Resumo"),
        "Taxonomia": st.session_state.get("aud_tax", None),
        "Pos_ESA": st.session_state.get("aud_pos_esa", None),
    }
    salvar_manifest(run_dir, cfg, inputs={"jpl_files": [p.name for p in jpl_paths]}, aud=aud_total)
    return aud_total


_init_status()


# =========================
# Sidebar: parâmetros
# =========================
st.sidebar.header("Parâmetros da Missão")

cfg = ConfigMissao(
    observatorio=st.sidebar.text_input("Observatório", "Y28"),
    data_inicio=st.sidebar.text_input("Data início (YYYY-MM-DD)", "2026-01-11"),
    data_fim=st.sidebar.text_input("Data fim (YYYY-MM-DD)", "2026-01-25"),

    # NOVO: hora início UTC (opcional)
    hora_inicio_utc=st.sidebar.text_input("Hora início UTC (HH:MM) (opcional)", "").strip() or None,

    step_min=st.sidebar.selectbox("Step (min)", [5, 10, 15, 20, 30, 60], index=1),
    ALT_MIN=st.sidebar.number_input("ALT_MIN (deg)", value=20.0, step=1.0),
    ALT_MAX=st.sidebar.number_input("ALT_MAX (deg)", value=70.0, step=1.0),
    V_MAX=st.sidebar.number_input("V_MAX", value=19.0, step=0.1),

    # NOVO: filtro por céu escuro (altura do Sol)
    SOL_ALT_MAX=st.sidebar.number_input(
        "SOL_ALT_MAX (deg) (opcional, céu escuro)",
        value=float("nan"),
        help="Use -18 (astronômico), -12 (náutico), -6 (civil). Deixe vazio para não filtrar."
    ),

    LIMIAR_RAPIDO=st.sidebar.number_input('Limiar rápido μ ("/min)', value=10.0, step=0.5),
    peso_recencia=st.sidebar.slider("Peso recência", 0.0, 1.0, 0.45, 0.05),
    peso_mag=st.sidebar.slider("Peso magnitude", 0.0, 1.0, 0.45, 0.05),
    peso_vel=0.0,  # ajustamos abaixo
)

# converte SOL_ALT_MAX: se NaN => None
try:
    if pd.isna(cfg.SOL_ALT_MAX):
        cfg.SOL_ALT_MAX = None
except Exception:
    cfg.SOL_ALT_MAX = None

cfg.peso_vel = max(0.0, 1.0 - (cfg.peso_recencia + cfg.peso_mag))
st.sidebar.caption(f"Peso velocidade ajustado automaticamente = {cfg.peso_vel:.2f}")

erros = validar_cfg(cfg)
if erros:
    st.sidebar.error("Config inválida:")
    for e in erros:
        st.sidebar.write(f"- {e}")
else:
    st.sidebar.success("Config OK")

st.sidebar.divider()
st.sidebar.header("Arquivos de entrada (JPL)")

uploaded = st.sidebar.file_uploader(
    "Envie 1 ou mais CSVs do JPL",
    type=["csv"],
    accept_multiple_files=True
)

st.sidebar.divider()
st.sidebar.header("Pós-ESA (opcional)")
post_esa_text = st.sidebar.text_area("Cole a lista pós-ESA (1 por linha)", height=160)


# =========================
# Estado do app (dados)
# =========================
if "run_dir" not in st.session_state:
    st.session_state.run_dir = None

for k in ["df_jpl", "lista_obj", "aud_jpl", "df_mpc_raw", "aud_mpc", "df_obs", "summary", "ranked", "aud_pos_esa"]:
    if k not in st.session_state:
        st.session_state[k] = None


# =========================
# Status cards sempre visíveis
# =========================
_render_status_cards()
st.divider()


# =========================
# Run folder
# =========================
colA, colB = st.columns([1, 1])

with colA:
    if st.button("Iniciar nova execução (run)"):
        if erros:
            st.error("Corrija a configuração antes de iniciar a execução.")
        else:
            run_dir = criar_run_dir(cfg)
            st.session_state.run_dir = run_dir
            st.session_state.status["run"] = str(run_dir)

            # limpa dados e status das etapas
            for k in ["df_jpl", "lista_obj", "aud_jpl", "df_mpc_raw", "aud_mpc", "df_obs", "summary", "ranked", "aud_pos_esa"]:
                st.session_state[k] = None

            for etapa in ["etapa1", "etapa2", "etapa3", "etapa4"]:
                st.session_state.status[etapa] = {"ok": False, "msg": "—", "aud": None}

            st.session_state.status["etapa1"]["msg"] = "Aguardando leitura JPL"
            st.session_state.status["etapa2"]["msg"] = "Aguardando MPC"
            st.session_state.status["etapa3"]["msg"] = "Aguardando filtros/ranking"
            st.session_state.status["etapa4"]["msg"] = "Aguardando Pós-ESA"

            st.success(f"Run criada: {run_dir}")

with colB:
    st.write("Run atual:", st.session_state.run_dir if st.session_state.run_dir else "—")

st.divider()


# =========================
# Etapa 1: JPL
# =========================
st.header("Etapa 1 — Ler e normalizar JPL")

if st.session_state.run_dir is None:
    st.info("Clique em **Iniciar nova execução (run)** para começar.")
    st.stop()

run_dir: Path = st.session_state.run_dir

jpl_paths = []
if uploaded:
    in_dir = run_dir / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    for f in uploaded:
        path = in_dir / f.name
        path.write_bytes(f.getbuffer())
        jpl_paths.append(path)

if st.button("Rodar leitura JPL"):
    if not jpl_paths:
        st.error("Envie pelo menos 1 CSV do JPL.")
    else:
        df_jpl, lista_obj, aud_jpl = ler_jpl_csvs(jpl_paths)

        st.session_state.df_jpl = df_jpl
        st.session_state.lista_obj = lista_obj
        st.session_state.aud_jpl = aud_jpl

        st.session_state.status["etapa1"]["ok"] = True
        st.session_state.status["etapa1"]["msg"] = f"OK — {len(lista_obj)} objetos normalizados"
        st.session_state.status["etapa1"]["aud"] = aud_jpl

        st.success("JPL lido e normalizado.")
        st.json(aud_jpl)
        st.write("Amostra objetos (até 20):")
        st.code("\n".join(lista_obj[:20]))
        st.dataframe(df_jpl.head(20), use_container_width=True)

st.divider()


# =========================
# Etapa 2: MPC automático + progresso real
# =========================
st.header("Etapa 2 — MPC automático (astroquery) + cache")

if st.session_state.lista_obj is None:
    st.info("Rode primeiro a leitura do JPL.")
    st.stop()

lista_obj = st.session_state.lista_obj
st.caption(f"Objetos para consultar no MPC: {len(lista_obj)}")

if st.button("Buscar efemérides MPC (astroquery)"):
    if erros:
        st.error("Config inválida. Corrija na sidebar antes de buscar MPC.")
    else:
        st.session_state.status["etapa2"]["msg"] = "Executando..."
        st.info("Iniciando consulta ao MPC via astroquery. Mostrando progresso por objeto (cache/baixa/falha).")

        bar = st.progress(0)
        status = st.empty()
        detail = st.empty()
        t0 = time.time()

        # callback que o pipeline chama a cada objeto/fase
        def progress_cb(i_atual: int, total: int, obj: str, fase: str):
            total = max(1, int(total))
            i_atual = int(i_atual)
            pct = int(round(100 * i_atual / total))
            bar.progress(max(0, min(100, pct)))
            status.info(f"Progresso: {pct}%  ({i_atual}/{total})")
            detail.caption(f"Objeto: {obj}  •  Fase: {fase}")

        df_mpc_raw, aud_mpc = obter_mpc_astroquery(lista_obj, cfg, run_dir, progress_cb=progress_cb)

        st.session_state.df_mpc_raw = df_mpc_raw
        st.session_state.aud_mpc = aud_mpc

        dt_s = round(time.time() - t0, 2)
        ok = (len(aud_mpc.get("falhas", [])) == 0) and (not df_mpc_raw.empty)

        st.session_state.status["etapa2"]["ok"] = bool(ok)
        st.session_state.status["etapa2"]["msg"] = f"Finalizado em {dt_s}s — linhas={len(df_mpc_raw)} | falhas={len(aud_mpc.get('falhas', []))}"
        st.session_state.status["etapa2"]["aud"] = aud_mpc

        st.success("Consulta MPC finalizada.")
        st.subheader("Auditoria MPC")
        st.json(aud_mpc)

        if aud_mpc.get("falhas"):
            st.warning(f"Houve {len(aud_mpc['falhas'])} falhas. Veja abaixo (primeiras 15):")
            st.dataframe(pd.DataFrame(aud_mpc["falhas"]).head(15), use_container_width=True)

        st.subheader("Prévia do dataframe MPC (primeiras 30 linhas)")
        st.dataframe(df_mpc_raw.head(30), use_container_width=True)

st.divider()


# =========================
# Etapa 3: Filtros + resumo + ranking
# =========================
st.header("Etapa 3 — Filtros, resumo e ranking")

if st.session_state.df_mpc_raw is None:
    st.info("Rode a etapa MPC para criar o dataframe.")
    st.stop()

df_mpc_raw = st.session_state.df_mpc_raw

if st.button("Filtrar → Classificar → Resumir → Ranqueiar"):
    df_obs, aud_filt = filtrar_epocas(df_mpc_raw, cfg)
    df_obs, classe_obj, aud_cls = classificar_velocidade(df_obs, cfg)
    summary, aud_sum = resumir_por_objeto(df_obs, cfg)
    ranked = ranquear(summary, cfg) if not summary.empty else summary

    st.session_state.df_obs = df_obs
    st.session_state.summary = summary
    st.session_state.ranked = ranked

    st.session_state.status["etapa3"]["ok"] = not ranked.empty
    st.session_state.status["etapa3"]["msg"] = f"OK — objetos={len(ranked) if hasattr(ranked, '__len__') else 0}"
    st.session_state.status["etapa3"]["aud"] = {"Filtros": aud_filt, "Classe": aud_cls, "Resumo": aud_sum}

    # Manifesto consolidado com auditoria atual
    aud_total = _salvar_manifest_atual(run_dir, cfg, jpl_paths)

    st.success("Pipeline executado. Manifesto salvo.")
    st.subheader("Auditoria")
    st.json(aud_total)

    st.subheader("Tabela ranqueada (prévia)")
    st.dataframe(ranked.head(50), use_container_width=True)

    out_dir = run_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    if not ranked.empty:
        csv_path = out_dir / "neos_tabela_geral.csv"
        ranked.to_csv(csv_path, sep=";", index=False)
        st.download_button("Baixar neos_tabela_geral.csv", data=csv_path.read_bytes(), file_name=csv_path.name)

st.divider()


# =========================
# Etapa 4 — Pós-ESA (implementada)
# =========================
st.header("Etapa 4 — Pós-ESA (opcional)")
st.caption("Cole a lista pós-ESA na sidebar para filtrar a tabela ranqueada e exportar a versão final.")

if st.session_state.ranked is None:
    st.info("Rode a Etapa 3 antes de usar Pós-ESA.")
else:
    ranked = st.session_state.ranked.copy()

    def _parse_lista_pos_esa(txt: str):
        if not txt:
            return []
        linhas = [x.strip() for x in txt.splitlines()]
        linhas = [x for x in linhas if x]
        # normalização básica
        out = []
        for s in linhas:
            s = s.replace("(", "").replace(")", "").strip()
            s = " ".join(s.split())
            out.append(s)
        return out

    pos_list = _parse_lista_pos_esa(post_esa_text)

    if not pos_list:
        st.info("Sem lista pós-ESA colada (ok).")
        st.session_state.status["etapa4"]["ok"] = False
        st.session_state.status["etapa4"]["msg"] = "Sem lista Pós-ESA"
    else:
        st.success(f"Lista Pós-ESA detectada: {len(pos_list)} objetos.")

        ranked["Nome_do_objeto_sem_asterisco"] = (
            ranked["Nome do objeto"].astype(str).str.replace("*", "", regex=False).str.strip()
        )
        ranked["Nome_limpo_norm"] = ranked["Nome_limpo"].astype(str).str.strip()

        mask = ranked["Nome_limpo_norm"].isin(pos_list) | ranked["Nome_do_objeto_sem_asterisco"].isin(pos_list)
        final_pos = ranked[mask].copy().reset_index(drop=True)

        encontrados = set(final_pos["Nome_limpo_norm"].tolist()) | set(final_pos["Nome_do_objeto_sem_asterisco"].tolist())
        nao_encontrados = [x for x in pos_list if x not in encontrados]

        aud_pos = {
            "pos_esa_itens": len(pos_list),
            "pos_esa_encontrados": int(len(final_pos)),
            "pos_esa_nao_encontrados": nao_encontrados[:80],
        }
        st.session_state.aud_pos_esa = aud_pos

        st.session_state.status["etapa4"]["ok"] = not final_pos.empty
        st.session_state.status["etapa4"]["msg"] = f"Encontrados {len(final_pos)}/{len(pos_list)}"
        st.session_state.status["etapa4"]["aud"] = aud_pos

        # NOVO: persistir manifesto após atualização pós-ESA
        _salvar_manifest_atual(run_dir, cfg, jpl_paths)

        st.subheader("Auditoria Pós-ESA")
        st.json(aud_pos)

        if final_pos.empty:
            st.warning("Nenhum item da lista Pós-ESA foi encontrado na tabela ranqueada.")
        else:
            final_show = final_pos.drop(columns=["Nome_do_objeto_sem_asterisco", "Nome_limpo_norm"], errors="ignore")

            st.subheader("Tabela final Pós-ESA (ranqueada)")
            st.dataframe(final_show, use_container_width=True)

            out_dir = run_dir / "outputs"
            out_dir.mkdir(exist_ok=True)
            csv_path = out_dir / "neos_tabela_pos_esa.csv"
            final_show.to_csv(csv_path, sep=";", index=False)

            st.download_button(
                "Baixar neos_tabela_pos_esa.csv",
                data=csv_path.read_bytes(),
                file_name=csv_path.name
            )
