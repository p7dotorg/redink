"""Generate a self-contained interactive HTML annotation report."""
import html as _html
import json
from pathlib import Path

_SEVERITY_ICON = {"critical": "●", "major": "!", "minor": "·"}


def generate(paper_text: str, findings: list, verdict, output_path: str | Path, title: str = "paper") -> None:
    """Write a self-contained interactive HTML annotation file."""
    annotated_body = _annotate_text(paper_text, findings)
    sidebar_html   = _sidebar(findings)
    findings_json  = json.dumps(_findings_data(findings))

    status   = getattr(verdict, "status", "?")
    summary  = _html.escape((getattr(verdict, "summary", "") or "")[:400])
    critical = getattr(verdict, "critical_count", 0)
    major    = getattr(verdict, "major_count", 0)
    minor    = getattr(verdict, "minor_count", 0)
    vcls     = {"PASS": "pass", "REVISE": "revise", "FAIL": "fail"}.get(status, "")

    n_crit = sum(1 for f in findings if getattr(f, "severity", "") == "critical")
    n_maj  = sum(1 for f in findings if getattr(f, "severity", "") == "major")
    n_min  = sum(1 for f in findings if getattr(f, "severity", "") == "minor")

    content = _TEMPLATE.format(
        title=_html.escape(title),
        verdict_status=status,
        vcls=vcls,
        summary=summary,
        critical=critical,
        major=major,
        minor=minor,
        n_all=len(findings),
        n_crit=n_crit,
        n_maj=n_maj,
        n_min=n_min,
        sidebar=sidebar_html,
        body=annotated_body,
        findings_json=findings_json,
    )
    Path(output_path).write_text(content, encoding="utf-8")


# ── text annotation ───────────────────────────────────────────────────────────

def _annotate_text(paper_text: str, findings: list) -> str:
    annotations: list[tuple[int, int, int, str]] = []
    lower = paper_text.lower()

    for i, f in enumerate(findings):
        evidence = (getattr(f, "evidence", "") or "").strip()
        if len(evidence) < 10:
            continue
        for length in (80, 50, 30, 15):
            snippet = evidence[:length].strip()
            if len(snippet) < 8:
                continue
            idx = lower.find(snippet.lower())
            if idx == -1:
                continue
            end = idx + len(snippet)
            if any(s < end and e > idx for (s, e, _, _) in annotations):
                continue
            annotations.append((idx, end, i, getattr(f, "severity", "minor")))
            break

    annotations.sort()
    parts, pos = [], 0
    for start, end, fid, sev in annotations:
        if start > pos:
            parts.append(_html.escape(paper_text[pos:start]))
        parts.append(f'<mark data-id="{fid}" class="m-{sev}">')
        parts.append(_html.escape(paper_text[start:end]))
        parts.append('</mark>')
        pos = end
    if pos < len(paper_text):
        parts.append(_html.escape(paper_text[pos:]))
    return "".join(parts)


# ── sidebar ───────────────────────────────────────────────────────────────────

def _sidebar(findings: list) -> str:
    items = []
    for i, f in enumerate(findings):
        sev      = getattr(f, "severity", "minor")
        icon     = _SEVERITY_ICON.get(sev, "·")
        dim      = _html.escape(getattr(f, "dimension", ""))
        persona  = _html.escape(getattr(f, "persona", ""))
        issue    = _html.escape((getattr(f, "issue", "") or "")[:160])
        fix      = _html.escape((getattr(f, "suggestion", "") or "")[:200])
        evidence = _html.escape((getattr(f, "evidence", "") or "")[:120])
        items.append(f"""
<div class="card s-{sev}" data-id="{i}" data-sev="{sev}">
  <div class="card-meta">
    <span class="badge {sev}">{icon} {sev.upper()}</span>
    <span class="tag">{dim}</span>
    <span class="tag dim">{persona}</span>
  </div>
  <div class="card-issue">{issue}</div>
  <div class="card-evidence">{evidence}</div>
  <div class="card-fix">↳ {fix}</div>
</div>""")
    return "\n".join(items)


def _findings_data(findings: list) -> list:
    return [
        {"id": i, "severity": getattr(f, "severity", "minor"),
         "dimension": getattr(f, "dimension", ""),
         "issue": (getattr(f, "issue", "") or "")[:200]}
        for i, f in enumerate(findings)
    ]


# ── HTML template ─────────────────────────────────────────────────────────────

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>redink — {title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --red:#E8252A;--red2:rgba(232,37,42,.18);--red3:rgba(232,37,42,.35);
  --bg:#0d0d0d;--panel:#141414;--border:#222;
  --text:#d8d8d8;--dim:#555;--dim2:#3a3a3a;
  --pass:#22c55e;--revise:#f59e0b;--fail:#E8252A;
}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);
      height:100vh;display:flex;flex-direction:column;overflow:hidden}}

/* ── header ── */
header{{
  display:flex;align-items:center;gap:16px;padding:0 24px;
  height:52px;border-bottom:1px solid var(--border);
  background:var(--panel);flex-shrink:0;
}}
.logo{{color:var(--red);font-weight:700;font-size:15px;letter-spacing:.05em}}
.verdict{{
  font-weight:700;font-size:13px;padding:3px 10px;border-radius:4px;
  letter-spacing:.08em;
}}
.verdict.pass{{color:#0d0d0d;background:var(--pass)}}
.verdict.revise{{color:#0d0d0d;background:var(--revise)}}
.verdict.fail{{color:#fff;background:var(--fail)}}
.counts{{display:flex;gap:12px;font-size:12px;margin-left:4px}}
.counts .c{{color:var(--red)}} .counts .m{{color:#f59e0b}} .counts .n{{color:#60a5fa}}
.summary{{font-size:11px;color:var(--dim);flex:1;
           overflow:hidden;white-space:nowrap;text-overflow:ellipsis}}

/* ── layout ── */
.layout{{display:flex;flex:1;overflow:hidden}}

/* ── sidebar ── */
aside{{
  width:320px;flex-shrink:0;display:flex;flex-direction:column;
  border-right:1px solid var(--border);background:var(--panel);
}}
.filters{{
  display:flex;gap:6px;padding:10px 12px;border-bottom:1px solid var(--border);
  flex-shrink:0;
}}
.filter{{
  font-size:11px;padding:3px 8px;border-radius:4px;border:1px solid var(--border);
  background:transparent;color:var(--dim);cursor:pointer;transition:.15s;
}}
.filter:hover,.filter.active{{background:var(--red2);color:var(--text);border-color:var(--red)}}
.filter[data-f=critical].active{{background:rgba(232,37,42,.25);border-color:var(--red)}}
.filter[data-f=major].active{{background:rgba(245,158,11,.15);border-color:#f59e0b;color:#f59e0b}}
.filter[data-f=minor].active{{background:rgba(96,165,250,.1);border-color:#60a5fa;color:#60a5fa}}
.count-badge{{
  font-size:10px;background:var(--dim2);padding:1px 5px;border-radius:10px;margin-left:3px;
}}

.cards{{flex:1;overflow-y:auto;padding:8px}}
.cards::-webkit-scrollbar{{width:4px}} .cards::-webkit-scrollbar-thumb{{background:var(--dim2)}}

.card{{
  padding:10px 12px;margin-bottom:6px;border-radius:6px;cursor:pointer;
  border:1px solid transparent;transition:.15s;
}}
.card:hover{{border-color:var(--border)}}
.card.active{{border-color:var(--red);background:var(--red2)}}
.card.hidden{{display:none}}

.card-meta{{display:flex;align-items:center;gap:6px;margin-bottom:5px}}
.badge{{font-size:10px;font-weight:700;padding:2px 6px;border-radius:3px;letter-spacing:.04em}}
.badge.critical{{color:var(--red);background:rgba(232,37,42,.15)}}
.badge.major{{color:#f59e0b;background:rgba(245,158,11,.12)}}
.badge.minor{{color:#60a5fa;background:rgba(96,165,250,.1)}}
.tag{{font-size:10px;color:var(--dim);background:var(--dim2);
      padding:1px 6px;border-radius:3px}}
.tag.dim{{background:transparent}}

.card-issue{{font-size:12px;color:var(--text);line-height:1.5;margin-bottom:4px}}
.card-evidence{{font-size:11px;color:var(--dim);font-style:italic;
                margin-bottom:4px;display:none}}
.card.active .card-evidence{{display:block}}
.card-fix{{font-size:11px;color:#60a5fa;display:none}}
.card.active .card-fix{{display:block}}

/* ── paper ── */
main{{flex:1;overflow-y:auto;padding:40px 56px;background:var(--bg)}}
main::-webkit-scrollbar{{width:6px}} main::-webkit-scrollbar-thumb{{background:var(--dim2)}}
pre{{
  font-family:'Courier New',Courier,monospace;font-size:13px;
  line-height:1.8;white-space:pre-wrap;word-break:break-word;color:#c8c8c8;
}}

/* ── marks ── */
mark{{
  background:transparent;border-radius:2px;padding:1px 0;
  transition:background .2s;cursor:pointer;
}}
mark.m-critical{{background:rgba(232,37,42,.22);border-bottom:2px solid var(--red)}}
mark.m-major{{background:rgba(245,158,11,.18);border-bottom:2px solid #f59e0b}}
mark.m-minor{{background:rgba(96,165,250,.12);border-bottom:2px solid #60a5fa}}
mark.pulse{{
  animation:pulse .6s ease;
}}
@keyframes pulse{{
  0%{{background:rgba(232,37,42,.5)}}
  100%{{background:rgba(232,37,42,.22)}}
}}
mark.m-major.pulse{{animation:pulse-maj .6s ease}}
@keyframes pulse-maj{{0%{{background:rgba(245,158,11,.5)}}100%{{background:rgba(245,158,11,.18)}}}}
mark.m-minor.pulse{{animation:pulse-min .6s ease}}
@keyframes pulse-min{{0%{{background:rgba(96,165,250,.4)}}100%{{background:rgba(96,165,250,.12)}}}}
</style>
</head>
<body>

<header>
  <span class="logo">redink</span>
  <span class="verdict {vcls}">{verdict_status}</span>
  <span class="counts">
    <span class="c">{critical} critical</span>
    <span class="m">{major} major</span>
    <span class="n">{minor} minor</span>
  </span>
  <span class="summary">{summary}</span>
</header>

<div class="layout">
  <aside>
    <div class="filters">
      <button class="filter active" data-f="all">ALL<span class="count-badge">{n_all}</span></button>
      <button class="filter" data-f="critical">● <span class="count-badge">{n_crit}</span></button>
      <button class="filter" data-f="major">! <span class="count-badge">{n_maj}</span></button>
      <button class="filter" data-f="minor">· <span class="count-badge">{n_min}</span></button>
    </div>
    <div class="cards" id="cards">
{sidebar}
    </div>
  </aside>

  <main id="paper">
    <pre id="text">{body}</pre>
  </main>
</div>

<script>
const findings = {findings_json};

// card click → scroll + pulse mark
document.querySelectorAll('.card').forEach(card => {{
  card.addEventListener('click', () => {{
    const id = card.dataset.id;
    document.querySelectorAll('.card.active').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    const mark = document.querySelector(`mark[data-id="${{id}}"]`);
    if (mark) {{
      mark.scrollIntoView({{behavior:'smooth', block:'center'}});
      mark.classList.remove('pulse');
      void mark.offsetWidth;
      mark.classList.add('pulse');
    }}
  }});
}});

// mark click → activate card
document.querySelectorAll('mark').forEach(mark => {{
  mark.addEventListener('click', () => {{
    const id = mark.dataset.id;
    const card = document.querySelector(`.card[data-id="${{id}}"]`);
    if (card) {{
      document.querySelectorAll('.card.active').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      card.scrollIntoView({{behavior:'smooth', block:'center'}});
    }}
  }});
}});

// filters
document.querySelectorAll('.filter').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.filter').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const f = btn.dataset.f;
    document.querySelectorAll('.card').forEach(c => {{
      c.classList.toggle('hidden', f !== 'all' && c.dataset.sev !== f);
    }});
    document.querySelectorAll('mark').forEach(m => {{
      const sev = m.className.replace('m-','').replace('pulse','').trim().split(' ')
                    .find(s => s.startsWith('m-'))?.replace('m-','');
      m.style.opacity = (f === 'all' || sev === f) ? '1' : '0.15';
    }});
  }});
}});
</script>
</body>
</html>
"""
