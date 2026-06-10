# NEO Mission Pipeline (Y28)

Aplicativo Streamlit para selecionar candidatos de NEOs para estudo de cores no Y28/OASI.

A lógica principal é:

1. ler uma lista inicial de objetos;
2. buscar efemérides no MPC;
3. avaliar as janelas noturnas de observação, por padrão das 21:00 às 08:00 UTC;
4. ranquear apenas bons candidatos observacionais;
5. consultar taxonomia publicada via ROCKS somente nos melhores candidatos;
6. exportar a lista de objetos sem taxonomia encontrada para Eddie levar ao coordenador.

O objetivo não é montar toda a tabela final da campanha. O objetivo é entregar uma lista limpa de objetos para cores, com informações suficientes para o coordenador incorporar esses alvos à tabela geral da missão.

---

## Instalação

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Como executar

```bash
streamlit run app.py
```

---

## Entrada

O app recebe um ou mais CSVs com nomes de objetos. A leitura procura automaticamente colunas como:

```text
Object Name, Object, Target, name, Name
```

Os nomes são normalizados removendo parênteses e espaços duplicados.

---

## Janela de observação

A campanha é avaliada por noite. Por padrão, cada noite vai de:

```text
21:00 UTC até 08:00 UTC do dia seguinte
```

Isso é importante porque altitude, magnitude, fase e velocidade aparente podem variar de uma noite para outra. Por isso o app gera uma tabela específica de janelas por noite.

---

## Fluxo no app

### Etapa 1 - Ler lista de objetos

Lê e normaliza os objetos enviados por CSV.

### Etapa 2 - Buscar efemérides MPC

Consulta o MPC para cobrir todas as noites da missão. Internamente, o app amplia a consulta até a manhã posterior à última noite para cobrir corretamente janelas 21:00-08:00 UTC.

### Etapa 3 - Avaliar janelas por noite

Filtra as efemérides por:

- magnitude aparente;
- altura mínima e máxima;
- altura do Sol, se configurada;
- janela noturna;
- duração mínima da janela.

Depois resume cada objeto por noite e classifica a qualidade da noite como:

```text
Boa
Regular
Ruim
```

### Etapa 4 - Verificar taxonomia publicada via ROCKS

Para reduzir o trabalho, o app consulta o ROCKS apenas nos melhores candidatos observacionais, e não em toda a lista inicial.

### Etapa 5 - Gerar produtos finais

Gera a lista principal de candidatos para estudo de cores e uma tabela auxiliar para o coordenador.

---

## Arquivos de saída

Os principais arquivos gerados em `runs/run_<timestamp>/outputs/` são:

```text
janelas_por_noite_eddie.csv
ranking_observacional_cores.csv
ranking_com_taxonomia_rocks.csv
candidatos_cores_eddie.csv
apoio_coordenador_eddie.csv
```

### `janelas_por_noite_eddie.csv`

Mostra a variação por noite:

- objeto;
- noite UTC;
- início e fim da janela;
- duração;
- magnitude inicial/final/mínima;
- altura inicial/final/máxima;
- fase inicial/final, quando disponível;
- velocidade angular;
- velocidades em AR e DEC, quando disponíveis;
- qualidade da noite.

### `ranking_observacional_cores.csv`

Mostra os melhores objetos observacionais antes da consulta de taxonomia.

### `ranking_com_taxonomia_rocks.csv`

Mostra o ranking observacional enriquecido com o resultado da consulta ROCKS.

### `candidatos_cores_eddie.csv`

Produto principal do projeto. Por padrão, contém apenas objetos sem taxonomia publicada encontrada.

### `apoio_coordenador_eddie.csv`

Tabela auxiliar com colunas próximas da tabela geral de campanha, incluindo:

```text
OBJETOS, D(Km), Prot(h), Porb(yr), H, Type, tp, Spectral, Albedo,
alpha_o, mo, v_ar_o, v_dec_o, alpha_f, mf, v_ar_f, v_dec_f,
Projects, Tipo_projeto, Filtros_sugeridos, Melhor_noite, Janela_UTC
```

Quando uma informação não é encontrada, o app preenche com `?`.

---

## Interpretação da taxonomia

A consulta ROCKS é usada para identificar estudos prévios de taxonomia/cores. A regra operacional adotada é:

```text
Sem taxonomia encontrada  -> Candidato para cores
Com taxonomia encontrada  -> Baixa prioridade/remover da lista principal
Consulta inconclusiva     -> Verificar manualmente
```

---

## Observações importantes

- O app prioriza reduzir trabalho: primeiro filtra bons objetos observáveis, depois consulta ROCKS.
- A tabela final da campanha continua sendo responsabilidade do coordenador.
- O app entrega os objetos de Eddie já organizados para estudo de cores.
- Algumas propriedades físicas, como diâmetro, albedo e período de rotação, podem não estar disponíveis. Nesses casos, a saída usa `?`.

---

## Próximos desenvolvimentos recomendados

- Verificar se o MPC retorna sempre as componentes de velocidade em AR e DEC para todos os objetos.
- Melhorar a extração de propriedades físicas do ROCKS conforme o formato real dos retornos.
- Incluir distância angular da Lua e fase lunar.
- Incluir estimativa de trailing por tempo de exposição.
- Adicionar um CSV de exemplo em `examples/`.
