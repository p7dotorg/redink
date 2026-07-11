# repro_check — verificação por execução (v1)

**Data:** 2026-07-10
**Status:** aprovado, pré-implementação

## Problema

Hoje o reviewer de `reproducibility` é heurística de texto: lê o paper e
_adivinha_ ("faltam detalhes pra reproduzir"). É chute educado. Qualquer um
cola o texto no ChatGPT e recebe o mesmo chute — não há moat.

A aposta de fase 2 mais diferenciadora do redink: **verificar por execução**.
Em vez de ler texto, o redink baixa o repo linkado do paper e **testa de
verdade**. O finding vira verificável — "clonei o código, não instala:
`ModuleNotFoundError: torch`" em vez de "a documentação parece incompleta".
Ninguém automatiza baixar-e-rodar o código de um paper; é o que responde
"por que pagar em vez de colar no ChatGPT?".

## Escopo v1 (deliberadamente estreito)

v1 faz **clone → instala → importa**, determinístico, **sem LLM**. Pega o caso
mais comum de repro quebrado (deps faltando, build quebra, import estoura).
Não roda testes, não reproduz números de tabela — isso é v2+ (agêntico,
precisa dados/GPU, marcado como teto nas notas de projeto).

O objetivo de v1 é **medir ganho de recall** contra a régua do `eval/`: provar
que executar > adivinhar antes de investir no caminho agêntico.

## Decisões travadas

| # | Decisão | Escolha |
|---|---------|---------|
| 1 | Escopo | Clona → instala → importa. Determinístico, sem LLM. |
| 2 | Isolamento | Docker local; rede cortada antes do import; sem secrets; fs efêmero. |
| 3 | URL do repo | Campo `code_repo` no `classify`; fallback pro input se já for repo. |
| 4 | Placement | `Send` extra no fan-out, paralelo aos reviewers. |
| 5 | Debate | Finding `grounded` bypassa debate; teto `major` (só `critical` se repo ausente). |
| 6 | Limites | timeout 180s · mem 2g · cpus 2 · pids-limit. |
| 7 | Ativação | opt-in via `REDINK_REPRO=1` (default off). |

## Arquitetura

### Fluxo no grafo

O pipeline atual é `fetch → classify → fan-out(reviewers×personas) → debate →
contradiction → blind_spot → judge → synthesize`. `repro_check` entra como um
`Send` extra no fan-out, **paralelo** aos reviewers de texto — o clone/install
no Docker (~30–120s) roda enquanto os reviewers rodam, escondendo a latência.

```
classify ──┬── Send(reviewer, dim×persona) ──┐
           ├── Send(figure_reviewer) ─────────┼── debate → contradiction → ...
           └── Send(repro_check) ─────────────┘
              (só se code_repo existe E reproducibility é dimensão E REDINK_REPRO=1)
```

`repro_check → debate` fecha o fan-out como os outros reviewers. Os findings
que ele emite entram em `state["findings"]` via `operator.add`, igual a todos.

### Componentes

**1. Schema (`schemas.py`)**

- `Classification.code_repo: Optional[str]` — URL do repo **oficial** do paper
  ("code/implementation available at..."), `null` se não houver. O LLM do
  classify distingue o repo do paper dos repos citados (baselines, datasets).
- `Finding.grounded: bool = False` — finding verificado por execução. Imune a
  debate e a dedup (um fato de execução não pode ser argumentado nem
  clusterizado pra fora).

**2. Novo módulo `repro.py`** (zero dependências novas — `subprocess` + CLI do Docker)

```python
@dataclass
class ReproResult:
    status: Literal["ok", "install_fail", "import_fail",
                    "repo_missing", "timeout", "no_docker"]
    repo_url: str
    package: Optional[str] = None   # pacote top-level que tentamos importar
    log: str = ""                   # cauda do stdout/stderr real

def docker_available() -> bool: ...

def run_repro_check(repo_url: str, *, timeout: int = 180,
                    mem: str = "2g", cpus: int = 2) -> ReproResult: ...
```

`run_repro_check` em duas fases, compartilhando um volume Docker efêmero:

- **Fase 1** (`--network=bridge`): `git clone` o repo + detecta o manifesto de
  deps (`requirements.txt` / `pyproject.toml` / `setup.py`) + instala num
  diretório-alvo no volume. `repo_missing` se o clone falha (404 / repo vazio).
- **Fase 2** (`--network=none`): resolve o pacote top-level e `import <pkg>`
  lendo do volume — o código do repo roda **sem rede**. `import_fail` se
  estoura.

Ambas as fases sob `--memory`, `--cpus`, `--pids-limit`, sem env/secrets
montados, container removido ao fim (`--rm`). Estouro de tempo global → `timeout`.

Imagem base: `python:3.11-slim`. (Versão de Python fixa é limitação conhecida —
repos que exigem outra versão podem dar `install_fail` "nosso"; por isso o teto
de severity é `major`, ver debate abaixo.)

**3. Node `repro_check` (`nodes_repro.py`)**

- Se `docker_available()` é falso → não emite finding (não pune o paper por
  lacuna de tooling nossa); grava aviso em `repro_result`.
- Chama `run_repro_check(code_repo)` e mapeia `ReproResult → Finding`:

  | status | severity | finding |
  |--------|----------|---------|
  | `repo_missing` | **critical** | "repo linkado não existe / está vazio" |
  | `install_fail` | major | "clonei, não instala: `<log>`" |
  | `import_fail` | major | "instala mas import quebra: `<log>`" |
  | `timeout` | major | "install/import passou de 180s" |
  | `ok` | — | **nenhum finding** |
  | `no_docker` | — | nenhum finding |

  Findings emitidos: `dimension="reproducibility"`, `grounded=True`,
  `evidence` = cauda do log real, `evidence_verified=True`.
- Grava `repro_result` no state **mesmo em `ok`** — pra o report/veredito poder
  afirmar "o código foi baixado e roda: instala e importa limpo". Metade do
  valor do moat é o sinal positivo, não só a falha.

**4. Grafo (`graph.py`)**

- `builder.add_node("repro_check", repro_check)`.
- `builder.add_edge("repro_check", "debate")`.
- `ReviewState` ganha `repro_result: Optional[dict]`.
- `route_to_reviewers`: se `os.getenv("REDINK_REPRO")` **e** `clf.code_repo`
  (ou `github_url` de input já é repo) **e** `"reproducibility" in clf.dimensions`
  → append `Send("repro_check", {"code_repo": <url>, "paper": state["paper"]})`.

**5. Debate bypass (`nodes_debate.py`)**

O `debate` só processa `severity == "critical"` (major/minor já passam
intocados). Duas mudanças cirúrgicas:

- Seleção de criticals exclui grounded:
  `criticals = [f for f in findings if f.severity == "critical" and not f.grounded]`
- Dedup roda só nos não-grounded; findings grounded são anexados verbatim
  depois do `_dedup_findings`.

Assim uma falha de repro verificada nunca é dismissada por um LLM defender.

**6. Ativação**

Opt-in via env `REDINK_REPRO=1` (default **off**). Rodar código arbitrário num
Docker a cada review é agressivo demais pra ligar sozinho. Ligado, dispara
sozinho quando há `code_repo` e `reproducibility` é dimensão.

## Estratégia de teste

- **Unit** (mock do `subprocess`): mapa `ReproResult → Finding` pra cada status;
  confirma severity, `grounded`, dimension.
- **Debate bypass**: um critical `grounded=True` sobrevive ao `debate`
  intocado (não vai pro defender, não é dismissado).
- **Route**: `REDINK_REPRO` + `code_repo` + dimensão → `Send("repro_check")`
  emitido; sem qualquer um dos três → não emitido.
- **Dedup**: finding grounded não é clusterizado com um finding de texto
  parecido; sai verbatim.
- **Integração** (gated em `docker_available()`, não roda em CI sem Docker):
  1 repo pequeno que instala+importa → `ok`; 1 repo com dep quebrada →
  `install_fail`.

## Fora de escopo (v2+)

- Rodar a suíte de testes / quickstart do repo (precisa LLM pra achar o comando).
- Reproduzir números de tabela (agêntico ReAct, precisa dados/GPU).
- Backend cloud (E2B/Modal) — o provider fica abstraído pra trocar depois, mas
  v1 é só Docker local.
- Múltiplas versões de Python / matriz de ambiente.

## Métrica de sucesso

Ganho de recall mensurável no `eval/`: findings de execução que a heurística de
texto não pega (ou pega com baixa confiança), sem introduzir falso-FAIL — o teto
`major` protege contra punir o paper por lacuna do harness.

## Addendum de validação (2026-07-10) — ESCOPO REAL

Rodado com Docker de verdade contra amostra do ASAP (ICLR 2018-2020). Números:

- **Cobertura:** 32% (95/300) dos papers linkam um repo GitHub real; 68% não têm
  código → repro_check pula (correto, sem finding).
- **Desfecho (amostra de 12 papers-com-repo, duas rodadas):** **0/12 `ok`.**
  Breakdown: install_fail 7 · no_official_repo 3 · repo_missing 1 · import_fail 1.

**Conclusão dura:** o modelo v1 (`pip install .` + `import`) **só produz `ok`
quando o produto do paper É um pacote pip** (ex: requests, umap-learn — ambos
deram `ok`). O paper de MÉTODO típico (ICLR) tem repo de scripts (`python
train.py`), sem `setup.py`/`requirements.txt` na raiz → `install_fail` (exit 30).

**Escopo honesto pra produção:** o `repro_check` v1 é valioso pra
papers-biblioteca/ferramenta/benchmark, **não** pra papers-método. Fica opt-in
(`REDINK_REPRO`, default off). Vender como "verifica papers que shippam pacote",
não "roda qualquer paper".

**O que destravaria o caso método (v2, não feito):** trocar o smoke-test de
"pip install" por "detecta entrypoint (`train.py`/comando do README) + instala
deps de onde estiverem + importa a fonte do repo dir sem exigir pacote". Isso é
inferência de deps/entrypoint (agêntico), não tweak. NÃO investir mais em parsing
de URL — é cauda longa que não move o `ok` (o fix de `resolve_repo_url` derrubou
`repo_missing` 3→1 mas manteve `ok` em 0).
