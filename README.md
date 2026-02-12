# NEO Mission Pipeline (Y28)

Aplicativo Streamlit para seleção e priorização de NEOs (Near-Earth Objects) para observação astronômica em filtros de cor.

---

## O que o projeto faz

A pipeline executa cinco etapas principais:

1. Leitura JPL  
2. Consulta automática ao MPC (astroquery)  
3. Filtros + classificação + ranking  
4. Enriquecimento taxonômico via ROCKS (opcional)  
5. Pós-ESA (opcional)

Cada execução salva auditoria completa em:

runs/run_<timestamp>/manifest.json

---

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
