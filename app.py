import time
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import (
    ConfigMissao,
    validar_cfg,
    criar_run_dir,
    salvar_manifest,
    ler_jpl_csvs,
)
from mpc_fix import obter_mpc_astroquery_resiliente
from rocks_fix import enriquecer_taxonomia_rocks_resiliente
from campaign import (
    validate_night_params,
    mpc_query_dates_for_nights,
    filter_observable_nights,
    summarize_by_night,
    rank_candidates,
    make_color_candidates,
    make_coordinator_support,
)
from diagnostics import cache_stats, clear_cache, rocks_diagnostic, identifiers_audit_df, environment_diagnostic

st.set_page_config(page_title="Y28 NEO Mission Pipeline", layout="wide")
st.title("Y28 - candidatos de NEOs para estudo de cores")
st.caption("Fluxo: melhores janelas noturnas primeiro; depois ROCKS para verificar taxonomia publicada.")


def init_state():
    keys = [
        "run_dir", "df_jpl", "lista_obj", "aud_jpl", "df_mpc_raw", "aud_mpc",
        "df_obs", "summary_night", "ranked", "ranked_tax", "candidatos",
        "apoio_coord", "aud_filt", "aud_night", "aud_rank", "aud_tax", "aud_final",
    ]
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None


init_state()

# =========================
# Sidebar
# =========================
st.sidebar.header("Parametros da missao")
obs = st.sidebar.text_input("Observatorio MPC", "Y28")
data_inicio = st.sidebar.text_input("Data inicio da campanha (YYYY-MM-DD)", "2026-01-11")
data_fim = st.sidebar.text_input("Data fim da campanha (YYYY-MM-DD)", "2026-01-25")

st.sidebar.subheader("Janela noturna UTC")
hora_noite_inicio = st.sidebar.text_input("Inicio da noite UTC", "21:00")
hora_noite_fim = st.sidebar.text_input("Fim da noite UTC", "08:00")
duracao_minima = st.sidebar.number_input("Duracao minima da janela (min)", value=60, min_value=10, step=10)

step_min = st.sidebar.selectbox("Passo das efemerides (min)", [5, 10, 15, 20, 30, 60], index=1)
ALT_MIN = st.sidebar.number_input("ALT_MIN (graus)", value=20.0, step=1.0)
ALT_MAX = st.sidebar.number_input("ALT_MAX (graus)", value=70.0, step=1.0)
V_MAX = st.sidebar.number_input("V_MAX", value=19.0, step=0.1)
SOL_ALT_MAX = st.sidebar.number_input("SOL_ALT_MAX opcional", value=float("nan"), help="Ex.: -18 para noite astronomica. Deixe vazio/NaN para nao filtrar.")
LIMIAR_RAPIDO = st.sidebar.number_input("Limiar rapido (arcsec/min)", value=10.0, step=0.5)
max_rocks = st.sidebar.number_input("Maximo de candidatos para consultar no ROCKS", value=50, min_value=1, step=5)
projeto_nome = st.sidebar.text_input("Nome no campo Projects", "Eddie")

cfg = ConfigMissao(
    observatorio=obs,
    data_inicio=data_inicio,
    data_fim=data_fim,
    hora_inicio_utc=None,
    step_min=int(step_min),
    ALT_MIN=float(ALT_MIN),
    ALT_MAX=float(ALT_MAX),
    V_MAX=float(V_MAX),
    SOL_ALT_MAX=None if pd.isna(SOL_ALT_MAX) else float(SOL_ALT_MAX),
    LIMIAR_RAPIDO=float(LIMIAR_RAPIDO),
)

cfg_mpc = cfg
try:
    q_ini, q_fim, q_hora = mpc_query_dates_for_nights(data_inicio, data_fim)
    cfg_mpc = ConfigMissao(
        observatorio=obs,
        data_inicio=q_ini,
        data_fim=q_fim,
        hora_inicio_utc=q_hora,
        step_min=int(step_min),
        ALT_MIN=float(ALT_MIN),
        ALT_MAX=float(ALT_MAX),
        V_MAX=float(V_MAX),
        SOL_ALT_MAX=None if pd.isna(SOL_ALT_MAX) else float(SOL_ALT_MAX),
        LIMIAR_RAPIDO=float(LIMIAR_RAPIDO),
    )
except Exception:
    pass

errors = validar_cfg(cfg) + validate_night_params(hora_noite_inicio, hora_noite_fim, int(duracao_minima))
if errors:
    st.sidebar.error("Configuracao invalida")
    for e in errors:
        st.sidebar.write(f"- {e}")
else:
    st.sidebar.success("Configuracao OK")

uploaded = st.sidebar.file_uploader("CSV(s) com objetos", type=["csv"], accept_multiple_files=True)

# =========================
# Ferramentas rápidas
# =========================
st.subheader("Ferramentas de controle")
ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
with ctrl1:
    st.metric("Arquivos no cache", cache_stats().get("parquet_files", 0))
with ctrl2:
    st.metric("Cache (MB)", cache_stats().get("total_mb", 0))
with ctrl3:
    if st.button("Apagar cache MPC local"):
        res = clear_cache()
        st.warning("Cache apagado. A proxima consulta MPC sera refeita do zero.")
        st.json(res)

env_diag = environment_diagnostic()
if env_diag.get("aiodns_instalado") or env_diag.get("pycares_instalado"):
    st.error("Ambiente com aiodns/pycares instalado. Isso pode quebrar o DNS do space-rocks no Windows.")
    st.code("pip uninstall aiodns pycares -y", language="powershell")
else:
    st.success("Ambiente ROCKS/DNS OK: aiodns e pycares não estão instalados.")

with st.expander("Diagnostico do reconhecimento de objetos e ROCKS", expanded=False):
    st.write("Diagnóstico do ambiente Python usado pelo app:")
    st.json(env_diag)
    exemplos_default = "1566 Icarus 1949 MA\n1036 Ganymed A924 UB\n398188 Agni 2010 LE15\n2021 VR3\n(23714) 1998 EC3"
    diag_text = st.text_area("Objetos para testar", value=exemplos_default, height=120)
    diag_objs = [x.strip() for x in diag_text.splitlines() if x.strip()]
    if diag_objs:
        st.write("Reconhecimento de identificadores:")
        st.dataframe(identifiers_audit_df(diag_objs), use_container_width=True)
    if st.button("Testar se ROCKS esta funcionando"):
        diag = rocks_diagnostic(diag_objs)
        if not diag.get("rocks_importado"):
            st.error("ROCKS/space-rocks nao foi importado no ambiente atual.")
        elif diag.get("aiodns_instalado") or diag.get("pycares_instalado"):
            st.error("ROCKS importado, mas aiodns/pycares ainda estão instalados. Remova antes de confiar no resultado.")
        else:
            st.success(f"ROCKS importado. Versao: {diag.get('rocks_version')}")
        st.json(diag)

# =========================
# Run
# =========================
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Iniciar nova execucao"):
        if errors:
            st.error("Corrija a configuracao antes de iniciar.")
        else:
            st.session_state.run_dir = criar_run_dir(cfg)
            for k in ["df_jpl", "lista_obj", "aud_jpl", "df_mpc_raw", "aud_mpc", "df_obs", "summary_night", "ranked", "ranked_tax", "candidatos", "apoio_coord", "aud_filt", "aud_night", "aud_tax", "aud_final"]:
                st.session_state[k] = None
            st.success(f"Run criada: {st.session_state.run_dir}")
with col2:
    st.write("Run atual:", st.session_state.run_dir if st.session_state.run_dir else "nenhuma")

if st.session_state.run_dir is None:
    st.info("Clique em 'Iniciar nova execucao' para comecar.")
    st.stop()

run_dir = Path(st.session_state.run_dir)

# =========================
# Etapa 1
# =========================
st.header("Etapa 1 - Ler lista de objetos")
jpl_paths = []
if uploaded:
    in_dir = run_dir / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    for f in uploaded:
        p = in_dir / f.name
        p.write_bytes(f.getbuffer())
        jpl_paths.append(p)

if st.button("Rodar leitura dos CSVs"):
    if not jpl_paths:
        st.error("Envie pelo menos um CSV.")
    else:
        df_jpl, lista_obj, aud_jpl = ler_jpl_csvs(jpl_paths)
        st.session_state.df_jpl = df_jpl
        st.session_state.lista_obj = lista_obj
        st.session_state.aud_jpl = aud_jpl
        st.success(f"{len(lista_obj)} objetos normalizados.")
        st.json(aud_jpl)
        st.write("Auditoria dos identificadores reconhecidos:")
        st.dataframe(identifiers_audit_df(lista_obj).head(100), use_container_width=True)
        st.dataframe(df_jpl.head(30), use_container_width=True)

if st.session_state.lista_obj is None:
    st.stop()

# =========================
# Etapa 2
# =========================
st.header("Etapa 2 - Buscar efemerides MPC")
st.caption(f"Consulta cobrindo noites {hora_noite_inicio}-{hora_noite_fim} UTC. Intervalo MPC usado: {cfg_mpc.data_inicio} {cfg_mpc.hora_inicio_utc} ate {cfg_mpc.data_fim} {cfg_mpc.hora_inicio_utc}.")
st.caption("Ordem de tentativa: numero oficial -> designacao provisoria -> designacao packed -> nome original.")

if st.button("Buscar efemerides"):
    if errors:
        st.error("Corrija a configuracao antes de consultar o MPC.")
    else:
        bar = st.progress(0)
        status = st.empty()
        t0 = time.time()

        def progress_cb(i_atual: int, total: int, obj: str, fase: str):
            pct = int(round(100 * int(i_atual) / max(1, int(total))))
            bar.progress(max(0, min(100, pct)))
            status.info(f"{pct}% ({i_atual}/{total}) - {obj} - {fase}")

        df_mpc_raw, aud_mpc = obter_mpc_astroquery_resiliente(st.session_state.lista_obj, cfg_mpc, run_dir, progress_cb=progress_cb)
        st.session_state.df_mpc_raw = df_mpc_raw
        st.session_state.aud_mpc = aud_mpc
        st.success(f"Consulta finalizada em {round(time.time() - t0, 1)} s. Linhas: {len(df_mpc_raw)}")
        st.json(aud_mpc)
        if aud_mpc.get("identificadores_usados"):
            st.write("Identificadores realmente usados no MPC:")
            st.dataframe(pd.DataFrame(aud_mpc["identificadores_usados"]).head(100), use_container_width=True)
        if aud_mpc.get("falhas"):
            st.warning(f"Falhas MPC: {len(aud_mpc['falhas'])}. Veja as primeiras abaixo.")
            st.dataframe(pd.DataFrame(aud_mpc["falhas"]).head(100), use_container_width=True)
        st.dataframe(df_mpc_raw.head(30), use_container_width=True)

if st.session_state.df_mpc_raw is None:
    st.stop()

# =========================
# Etapa 3
# =========================
st.header("Etapa 3 - Avaliar janelas por noite e ranquear bons candidatos")

if st.button("Filtrar por noite e ranquear"):
    df_obs, aud_filt = filter_observable_nights(st.session_state.df_mpc_raw, cfg, hora_noite_inicio, hora_noite_fim)
    summary_night, aud_night = summarize_by_night(df_obs, cfg, int(duracao_minima))
    ranked = rank_candidates(summary_night, int(duracao_minima))

    st.session_state.df_obs = df_obs
    st.session_state.summary_night = summary_night
    st.session_state.ranked = ranked
    st.session_state.aud_filt = aud_filt
    st.session_state.aud_night = aud_night

    out_dir = run_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    if not summary_night.empty:
        summary_night.to_csv(out_dir / "janelas_por_noite_eddie.csv", sep=";", index=False)
    if not ranked.empty:
        ranked.to_csv(out_dir / "ranking_observacional_cores.csv", sep=";", index=False)

    salvar_manifest(run_dir, cfg, inputs={"jpl_files": [p.name for p in jpl_paths]}, aud={"JPL": st.session_state.aud_jpl, "MPC": st.session_state.aud_mpc, "Filtro_noturno": aud_filt, "Resumo_noite": aud_night})

    st.success(f"Candidatos observacionais: {len(ranked)}")
    st.subheader("Resumo por noite")
    st.dataframe(summary_night.head(100), use_container_width=True)
    st.subheader("Ranking observacional")
    st.dataframe(ranked.head(50), use_container_width=True)

if st.session_state.ranked is None:
    st.stop()

# =========================
# Etapa 4
# =========================
st.header("Etapa 4 - Verificar taxonomia publicada via ROCKS")
st.caption("ROCKS agora usa o mesmo identificador preferido: numero oficial primeiro; designacao provisoria depois.")

if st.button("Consultar ROCKS nos melhores candidatos"):
    base = st.session_state.ranked.head(int(max_rocks)).copy()
    tax_status = st.empty()

    def tax_progress_cb(i_atual: int, total: int, obj: str, fase: str):
        tax_status.info(f"ROCKS {i_atual}/{total}: {obj}")

    ranked_tax, aud_tax = enriquecer_taxonomia_rocks_resiliente(base, progress_cb=tax_progress_cb)
    if "Taxonomia disponível" in ranked_tax.columns:
        ranked_tax["Taxonomia_encontrada"] = ranked_tax["Taxonomia disponível"]
    if "Classe taxonômica" in ranked_tax.columns:
        ranked_tax["Classe_taxonomica"] = ranked_tax["Classe taxonômica"]
    if "Fonte taxonomia" in ranked_tax.columns:
        ranked_tax["Fonte_taxonomia"] = ranked_tax["Fonte taxonomia"]

    st.session_state.ranked_tax = ranked_tax
    st.session_state.aud_tax = aud_tax

    out_dir = run_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    if not ranked_tax.empty:
        ranked_tax.to_csv(out_dir / "ranking_com_taxonomia_rocks.csv", sep=";", index=False)

    st.success("Consulta ROCKS concluida.")
    st.json(aud_tax)
    if aud_tax.get("falhas"):
        st.warning("Algumas consultas ROCKS falharam ou foram inconclusivas.")
        st.dataframe(pd.DataFrame(aud_tax["falhas"]).head(100), use_container_width=True)
    st.dataframe(ranked_tax.head(50), use_container_width=True)

if st.session_state.ranked_tax is None:
    st.stop()

# =========================
# Etapa 5
# =========================
st.header("Etapa 5 - Gerar produtos finais para Eddie e coordenador")
apenas_sem_tax = st.checkbox("Exportar lista principal apenas com objetos sem taxonomia encontrada", value=True)

if st.button("Gerar produtos finais"):
    candidatos, aud_final = make_color_candidates(st.session_state.ranked_tax, only_without_taxonomy=apenas_sem_tax)
    apoio = make_coordinator_support(candidatos, project=projeto_nome)

    st.session_state.candidatos = candidatos
    st.session_state.apoio_coord = apoio
    st.session_state.aud_final = aud_final

    out_dir = run_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    candidatos_path = out_dir / "candidatos_cores_eddie.csv"
    apoio_path = out_dir / "apoio_coordenador_eddie.csv"
    candidatos.to_csv(candidatos_path, sep=";", index=False)
    apoio.to_csv(apoio_path, sep=";", index=False)

    salvar_manifest(run_dir, cfg, inputs={"jpl_files": [p.name for p in jpl_paths]}, aud={"JPL": st.session_state.aud_jpl, "MPC": st.session_state.aud_mpc, "Filtro_noturno": st.session_state.aud_filt, "Resumo_noite": st.session_state.aud_night, "Taxonomia": st.session_state.aud_tax, "Final": aud_final})

    st.success(f"Produtos gerados. Candidatos finais: {len(candidatos)}")
    st.json(aud_final)

    st.subheader("candidatos_cores_eddie.csv")
    st.dataframe(candidatos, use_container_width=True)
    st.download_button("Baixar candidatos_cores_eddie.csv", data=candidatos_path.read_bytes(), file_name="candidatos_cores_eddie.csv")

    st.subheader("apoio_coordenador_eddie.csv")
    st.dataframe(apoio, use_container_width=True)
    st.download_button("Baixar apoio_coordenador_eddie.csv", data=apoio_path.read_bytes(), file_name="apoio_coordenador_eddie.csv")
