# NEO Mission Pipeline (Y28)

Aplicativo Streamlit para seleção e priorização de NEOs (Near-Earth Objects) para planejamento de observações astronômicas pelo Y28/OASI.

O projeto foi pensado para apoiar a escolha de alvos observáveis em janelas específicas, com base em efemérides, magnitude aparente, altitude, velocidade angular e critérios operacionais de missão.

---

## O que o projeto faz

A pipeline executa cinco blocos principais:

1. **Leitura JPL**  
   Lê um ou mais CSVs exportados do JPL e normaliza os nomes dos objetos.

2. **Consulta automática ao MPC**  
   Usa `astroquery.mpc` para consultar efemérides dos objetos para o observatório definido, por padrão `Y28`.

3. **Filtros + classificação + ranking**  
   Filtra as efemérides por magnitude, altitude, altitude solar opcional e velocidade angular. Em seguida, resume os objetos e gera uma tabela ranqueada.

4. **Taxonomia publicada via ROCKS (opcional)**  
   Consulta informações taxonômicas publicadas usando o pacote `rocks`, quando disponível no ambiente.

5. **Pós-ESA (opcional)**  
   Permite cruzar a tabela ranqueada com uma lista final de objetos pós-seleção externa.

Cada execução salva auditoria completa em:

```text
runs/run_<timestamp>/manifest.json
```

---

## Instalação

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Como executar

Na pasta raiz do projeto, rode:

```bash
streamlit run app.py
```

O navegador abrirá a interface do aplicativo.

---

## Arquivos de entrada

O app espera um ou mais arquivos CSV exportados do JPL.

A pipeline procura automaticamente uma coluna de nome do objeto entre as seguintes opções:

```text
Object Name, Object, Target, name, Name
```

Os nomes são normalizados removendo parênteses e espaços duplicados. Exemplos:

```text
(433) Eros  ->  433 Eros
2024 AB     ->  2024 AB
```

---

## Parâmetros principais

Na barra lateral do app é possível configurar:

- **Observatório**: código MPC do observatório. O padrão é `Y28`.
- **Data início / Data fim**: intervalo da missão em formato `YYYY-MM-DD`.
- **Hora início UTC**: opcional, em formato `HH:MM`.
- **Step (min)**: passo temporal das efemérides.
- **ALT_MIN / ALT_MAX**: limites de altitude do objeto em graus.
- **V_MAX**: magnitude aparente máxima permitida.
- **SOL_ALT_MAX**: filtro opcional para céu escuro. Exemplos: `-18` para noite astronômica, `-12` para noite náutica e `-6` para noite civil.
- **Limiar rápido**: velocidade angular em arcsec/min usada para classificar objetos como lentos ou rápidos.
- **Pesos do ranking**: controle da importância relativa de recência, magnitude e velocidade.

---

## Saídas geradas

Cada execução cria uma pasta em:

```text
runs/run_<timestamp>/
```

Com subpastas:

```text
inputs/
outputs/
logs/
```

Os principais arquivos de saída são:

```text
outputs/neos_tabela_geral.csv
outputs/neos_tabela_geral_taxonomia.csv
outputs/neos_tabela_pos_esa.csv
manifest.json
```

Nem todos os arquivos serão gerados em todas as execuções. Por exemplo, a tabela com taxonomia só é criada quando a etapa ROCKS é executada.

---

## Como o ranking funciona hoje

O ranking atual usa três componentes principais:

- **Magnitude**: objetos mais brilhantes recebem maior prioridade.
- **Velocidade angular**: objetos mais lentos recebem maior prioridade, pois tendem a ser mais fáceis de acompanhar e menos sujeitos a trailing.
- **Recência**: objetos com melhor janela mais próxima do fim do intervalo recebem maior peso, conforme o critério implementado.

O score final é calculado por uma soma ponderada:

```text
score_total = peso_recencia * score_recencia
            + peso_mag      * score_mag
            + peso_vel      * score_vel
```

---

## Observação científica importante

No estado atual, o app deve ser entendido principalmente como uma pipeline de **priorização observacional de NEOs**.

Para se tornar uma pipeline completa de **seleção para fotometria multibanda em cores**, ainda é recomendável incorporar critérios adicionais, como:

- tempo necessário para sequência Sloan, por exemplo `g' r' i' r' z' r'`;
- janela mínima para completar a sequência multibanda;
- estimativa de trailing em função da velocidade angular e do tempo de exposição;
- relação sinal-ruído esperada por filtro;
- distância angular da Lua e fase lunar;
- cálculo posterior de cores, refletância relativa e classificação taxonômica fotométrica.

---

## Taxonomia

A etapa ROCKS consulta taxonomias já publicadas, quando disponíveis. Portanto, essa etapa não deve ser confundida com classificação taxonômica obtida a partir das cores observadas pelo próprio projeto.

Recomenda-se distinguir:

1. **Taxonomia publicada via ROCKS**: informação externa já existente.
2. **Taxonomia fotométrica futura**: classificação inferida a partir de cores Sloan medidas pelo projeto.

---

## Próximos desenvolvimentos recomendados

- Melhorar a leitura dos CSVs para aceitar automaticamente `,` e `;`.
- Validar se `data_fim` é posterior ou igual a `data_inicio`.
- Incluir score de janela observacional usando `n_epocas` ou duração da janela.
- Incluir score de altitude usando `ALT_med` ou `ALT_max`.
- Registrar de forma mais detalhada os arquivos gerados no `manifest.json`.
- Criar uma pasta `examples/` com CSV de teste.
- Implementar módulo específico para viabilidade de sequência Sloan.
