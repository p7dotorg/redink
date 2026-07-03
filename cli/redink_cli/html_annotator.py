"""Generate a self-contained interactive HTML annotation report.

Layout: white paper column + right-margin notes (Google Docs style).
On mobile the notes collapse inline below each highlight.
"""
import html as _html
import json
import re
from pathlib import Path

_SEVERITY_ICON  = {"critical": "●", "major": "!", "minor": "·"}
_SEVERITY_COLOR = {
    "critical": ("#E8252A", "rgba(232,37,42,.12)", "rgba(232,37,42,.9)"),
    "major":    ("#d97706", "rgba(217,119,6,.10)",  "rgba(217,119,6,.9)"),
    "minor":    ("#2563eb", "rgba(37,99,235,.08)",   "rgba(37,99,235,.9)"),
}


def generate(
    paper_text: str,
    findings: list,
    verdict,
    output_path: str | Path,
    title: str = "paper",
) -> None:
    """Write a self-contained interactive HTML file."""
    title_parsed, authors, abstract, body_text = _parse_header(paper_text)
    if title_parsed:
        annotated, note_meta = _annotate(body_text, findings)
        header_html = _render_academic_header(title_parsed, authors, abstract)
    else:
        annotated, note_meta = _annotate(paper_text, findings)
        header_html = ''

    status   = getattr(verdict, "status", "?")
    summary  = _html.escape((getattr(verdict, "summary", "") or "")[:300])
    critical = getattr(verdict, "critical_count", 0)
    major    = getattr(verdict, "major_count", 0)
    minor    = getattr(verdict, "minor_count", 0)
    vcls     = {"PASS": "pass", "REVISE": "revise", "FAIL": "fail"}.get(status, "")

    n = {s: sum(1 for f in findings if getattr(f, "severity", "") == s)
         for s in ("critical", "major", "minor")}

    content = _TEMPLATE.format(
        title    = _html.escape(title),
        vcls     = vcls,
        vstatus  = status,
        summary  = summary,
        critical = critical,
        major    = major,
        minor    = minor,
        n_all    = len(findings),
        n_crit   = n["critical"],
        n_maj    = n["major"],
        n_min    = n["minor"],
        header   = header_html,
        body     = annotated,
        meta_json= json.dumps(note_meta),
    )
    Path(output_path).write_text(content, encoding="utf-8")


# ── text helpers ──────────────────────────────────────────────────────────────

def _txt(s: str) -> str:
    """Escape plain text and convert newlines to HTML spacing."""
    s = _html.escape(s)
    s = re.sub(r'\n{2,}', '<br><br>', s)
    s = s.replace('\n', ' ')
    return s


def _parse_header(text: str) -> tuple[str, list[dict], str, str]:
    """Parse arXiv plain text → (title, authors, abstract, body).

    Returns ('', [], '', text) if structure not detected.
    """
    and_pos = text.find('\\AND ')
    if and_pos == -1:
        return '', [], '', text

    pre_and = text[:and_pos]
    rest    = text[and_pos + 5:]

    # Title: last meaningful line before \\AND
    title = ''
    skip = {'arXiv', '[', 'Permission', 'permission', 'Copyright', 'Provided'}
    for line in reversed(pre_and.split('\n')):
        line = line.strip()
        if len(line) > 10 and not any(line.startswith(s) for s in skip):
            title = line
            break

    # Where does author section end → body begins?
    body_start = len(rest)
    for kw in ['Abstract\n', 'Abstract \n', '\nAbstract ', ' Abstract ',
               '\n1 Introduction', '1 Introduction ', '\n1\n']:
        idx = rest.find(kw)
        if idx != -1 and idx < body_start:
            body_start = idx

    author_section = rest[:body_start]
    body_text      = rest[body_start:]

    # Parse individual author blocks (separated by & / &amp;)
    raw_blocks = re.split(r'\n\s*(?:&amp;|&(?![a-zA-Z#;]))', author_section)
    authors = []
    for block in raw_blocks:
        block = re.sub(r'\d+\s*footnotemark:\s*\d+', '', block)
        block = re.sub(r'\s+\d+(?:\s+\d+)*\s*$', '', block, flags=re.MULTILINE)
        block = re.sub(r'(?<=[a-zA-Z])\s+\d+\s*$', '', block, flags=re.MULTILINE)
        block = block.replace('\xa0', ' ')
        lines = [l.strip() for l in block.split('\n') if l.strip() and len(l.strip()) > 1]
        if not lines:
            continue
        name = lines[0].strip()
        if len(name) > 60 or name.lower().startswith('equal') or '@' in name:
            continue
        # Only consider up to 4 lines after name; stop at long prose (footnotes)
        candidate_lines = []
        for l in lines[1:5]:
            if len(l) > 80 or l.lower().startswith('equal') or l.lower().startswith('ashish') or l.lower().startswith('noam'):
                break
            candidate_lines.append(l)
        affil = next((l for l in candidate_lines if '@' not in l), '')
        email = next((l for l in candidate_lines if '@' in l), '')
        if len(name) > 2:
            authors.append({'name': name, 'affil': affil, 'email': email})

    # Extract abstract (look for "Abstract" keyword then capture until next section)
    abstract = ''
    abs_m = re.search(r'Abstract\s*\n?(.*?)(?=\n\s*\n\s*\d+\s+\w|\Z)', body_text[:3000], re.DOTALL)
    if abs_m:
        abstract = re.sub(r'\s+', ' ', abs_m.group(1)).strip()
        body_text = body_text[abs_m.end():]

    return title, authors, abstract, body_text.strip()


def _render_academic_header(title: str, authors: list[dict], abstract: str) -> str:
    """Render an academic-paper-style header block."""
    title_html = _html.escape(title)

    if authors:
        cols = min(3, len(authors))
        author_cells = ''.join(
            f'<div class="ph-author">'
            f'<div class="ph-name">{_html.escape(a["name"])}</div>'
            f'<div class="ph-affil">{_html.escape(a["affil"])}</div>'
            f'<div class="ph-email">{_html.escape(a["email"])}</div>'
            f'</div>'
            for a in authors
        )
        authors_html = f'<div class="ph-authors" style="--ph-cols:{cols}">{author_cells}</div>'
    else:
        authors_html = ''

    abstract_html = (
        f'<div class="ph-abstract"><span class="ph-abs-label">Abstract</span>'
        f' {_html.escape(abstract)}</div>'
    ) if abstract else ''

    return (
        f'<div class="paper-header">'
        f'<h1 class="ph-title">{title_html}</h1>'
        f'{authors_html}'
        f'{abstract_html}'
        f'<hr class="ph-rule">'
        f'</div>'
    )


# ── annotation ────────────────────────────────────────────────────────────────

def _annotate(paper_text: str, findings: list) -> tuple[str, list]:
    """Return (html_body, note_meta_list).

    Each mark is immediately followed by its inline note.
    On desktop the note is CSS-floated into the right margin;
    on mobile it collapses inline below the highlight.
    """
    hits: list[tuple[int, int, int, object]] = []
    lower = paper_text.lower()

    for i, f in enumerate(findings):
        ev = (getattr(f, "evidence", "") or "").strip()
        for length in (80, 55, 35, 15):
            snippet = ev[:length].strip()
            if len(snippet) < 8:
                continue
            idx = lower.find(snippet.lower())
            if idx == -1:
                continue
            end = idx + len(snippet)
            if any(s < end and e > idx for (s, e, _, _) in hits):
                continue
            hits.append((idx, end, i, f))
            break

    hits.sort()

    parts, pos = [], 0
    meta = []

    for start, end, fid, f in hits:
        sev   = getattr(f, "severity", "minor")
        col, bg, _ = _SEVERITY_COLOR.get(sev, _SEVERITY_COLOR["minor"])
        icon  = _SEVERITY_ICON.get(sev, "·")
        dim   = _html.escape(getattr(f, "dimension", ""))
        per   = _html.escape(getattr(f, "persona", ""))
        issue = _html.escape((getattr(f, "issue", "") or "")[:200])
        fix   = _html.escape((getattr(f, "suggestion", "") or "")[:180])

        if start > pos:
            parts.append(_txt(paper_text[pos:start]))

        # Highlighted text
        parts.append(
            f'<mark class="hl hl-{sev}" data-id="{fid}" '
            f'style="background:{bg};border-bottom:2px solid {col}">'
            f'{_txt(paper_text[start:end])}</mark>'
        )

        # Margin note + mobile tap-indicator after the mark (no leading newline)
        parts.append(
            f'<sup class="hl-tag" data-id="{fid}" style="color:{col}" title="tap for note">{icon}</sup>'
            f'<aside class="note note-{sev}" data-id="{fid}" data-sev="{sev}" style="--nc:{col}">'
            f'<div class="note-header">'
            f'<span class="note-badge" style="color:{col}">{icon} {sev.upper()}</span>'
            f'<span class="note-tags">{dim} · {per}</span>'
            f'</div>'
            f'<div class="note-issue">{issue}</div>'
            f'<div class="note-fix">↳ {fix}</div>'
            f'</aside>'
        )

        meta.append({"id": fid, "severity": sev})
        pos = end

    if pos < len(paper_text):
        parts.append(_txt(paper_text[pos:]))

    return "".join(parts), meta


# ── template ──────────────────────────────────────────────────────────────────

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>redink — {title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400&family=Lora:ital,wght@0,400;0,700;1,400&family=Inter:wght@400;700&display=swap" rel="stylesheet">
<script>
MathJax = {{
  tex: {{inlineMath: [['$','$'],['\\\\(','\\\\)']]}},
  options: {{skipHtmlTags: ['script','style','pre','code']}},
  startup: {{typeset: false}}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  /* WIRED design tokens */
  --ink:#000000;
  --ink-soft:#1a1a1a;
  --canvas:#ffffff;
  --canvas-soft:#f5f5f5;
  --hairline:#e0e0e0;
  --body-col:#757575;

  /* redink severity (functional, not decorative) */
  --red:#E8252A;
  --amber:#d97706;
  --blue:#2563eb;
}}

body{{
  font-family:'Inter','Helvetica Neue',Arial,sans-serif;
  background:var(--canvas-soft);
  color:var(--ink);
  min-height:100vh;
}}

/* ── masthead band ── */
.masthead{{
  background:var(--canvas);
  border-bottom:2px solid var(--ink);
  padding:0 24px;
  display:flex;align-items:stretch;gap:0;
  position:sticky;top:0;z-index:100;
}}
.masthead-logo{{
  font-family:'Playfair Display','Georgia',serif;
  font-weight:400;font-size:22px;
  letter-spacing:-0.3px;color:var(--ink);
  padding:8px 20px 8px 0;
  border-right:1px solid var(--hairline);
  display:flex;align-items:center;gap:8px;
  text-decoration:none;
}}
.masthead-logo svg{{flex-shrink:0}}
/* "red" in brand red, "ink" in black */
.masthead-logo .brand-red{{color:var(--red)}}
.masthead-meta{{
  display:flex;align-items:stretch;gap:0;
  flex:1;min-width:0;
}}
.vbadge{{
  font-family:'Inter',sans-serif;font-size:10px;font-weight:700;
  letter-spacing:0.5px;text-transform:uppercase;
  padding:0 12px;
  border:none;border-right:1px solid var(--hairline);
  border-radius:0;
  background:var(--ink);color:var(--canvas);
  white-space:nowrap;display:flex;align-items:center;
}}
.vbadge.pass{{background:#16a34a}}
.vbadge.revise{{background:var(--amber)}}
.vbadge.fail{{background:var(--red)}}

/* severity filter buttons — live in masthead */
.sev-btn{{
  font-family:'Inter',sans-serif;font-size:11px;font-weight:700;
  padding:0 14px;
  border:none;border-right:1px solid var(--hairline);
  border-bottom:3px solid transparent;   /* indicator line */
  border-radius:0;
  background:transparent;cursor:pointer;
  color:var(--body-col);
  display:flex;align-items:center;gap:4px;
  transition:color .12s,border-bottom-color .12s,background .12s;
  white-space:nowrap;
}}
.sev-btn:hover{{background:var(--canvas-soft)}}

/* active: filled pill-less badge style */
.sev-btn[data-f="critical"].on{{color:var(--red);border-bottom-color:var(--red)}}
.sev-btn[data-f="major"].on{{color:var(--amber);border-bottom-color:var(--amber)}}
.sev-btn[data-f="minor"].on{{color:var(--blue);border-bottom-color:var(--blue)}}
.sev-btn[data-f="all"].on{{color:var(--ink);border-bottom-color:var(--ink);background:var(--canvas-soft)}}

.masthead-summary{{
  font-family:'Inter',sans-serif;font-size:11px;color:var(--body-col);
  flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
  display:flex;align-items:center;padding:0 16px;
}}

/* ── navigation — in masthead ── */
.nav{{
  display:flex;align-items:stretch;gap:0;
}}
.nav-btn{{
  font-family:'Inter',sans-serif;font-size:13px;
  padding:0 14px;
  border:none;border-left:1px solid var(--hairline);
  border-radius:0;
  background:transparent;cursor:pointer;color:var(--body-col);
  transition:color .12s,background .12s;line-height:1;
  display:flex;align-items:center;
}}
.nav-btn:hover:not(:disabled){{background:var(--canvas-soft);color:var(--ink)}}
.nav-btn:disabled{{opacity:.3;cursor:default}}
.nav-pos{{
  font-family:'Inter',sans-serif;font-size:11px;
  color:var(--body-col);padding:0 12px;
  border-left:1px solid var(--hairline);
  white-space:nowrap;display:flex;align-items:center;
}}

/* ── paper + margin layout ── */
.outer{{
  max-width:980px;
  margin:40px auto;
  padding:0 24px;
}}
.paper{{
  background:var(--canvas);
  border:1px solid var(--hairline);
  border-top:2px solid var(--ink);       /* WIRED: heavy top rule */
  padding:56px 64px 80px 64px;
  padding-right:304px;
  position:relative;
}}
.paper-eyebrow{{
  font-family:'Inter',sans-serif;
  font-size:11px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;
  color:var(--body-col);
  padding-bottom:12px;
  border-bottom:1px solid var(--hairline);
  margin-bottom:28px;
}}

/* ── academic paper header ── */
.paper-header{{
  text-align:center;
  margin-bottom:40px;
  /* extend into the right padding zone so it spans the full paper card */
  margin-right:-240px;   /* cancels the extra right padding (304-64=240) */
}}
.ph-title{{
  font-family:'Lora','Georgia',serif;
  font-size:26px;font-weight:700;line-height:1.25;
  color:var(--ink);
  margin-bottom:24px;
}}
.ph-authors{{
  display:grid;
  grid-template-columns:repeat(var(--ph-cols,3),1fr);
  gap:12px 24px;
  margin-bottom:20px;
}}
.ph-author{{text-align:center}}
.ph-name{{font-family:'Lora',serif;font-size:14px;font-weight:700;color:var(--ink)}}
.ph-affil{{font-family:'Inter',sans-serif;font-size:12px;color:var(--body-col);margin-top:2px}}
.ph-email{{font-family:'Inter',sans-serif;font-size:11px;color:var(--body-col);margin-top:1px}}
.ph-abstract{{
  text-align:left;
  font-family:'Lora',serif;font-size:14px;line-height:1.7;
  color:var(--ink-soft);
  margin:20px 0 0;
  padding:16px 20px;
  background:var(--canvas-soft);
  border-left:3px solid var(--hairline);
}}
.ph-abs-label{{font-family:'Inter',sans-serif;font-weight:700;font-size:12px;letter-spacing:.4px;text-transform:uppercase}}
.ph-rule{{border:none;border-top:1px solid var(--hairline);margin:28px 0}}
@media(max-width:860px){{
  .paper-header{{margin-right:0}}
  .ph-authors{{grid-template-columns:repeat(2,1fr)}}
  .ph-title{{font-size:20px}}
}}
.paper-text{{
  font-family:'Lora','Georgia',serif;
  font-size:16px;line-height:1.75;
  word-break:break-word;
  color:var(--ink);
  position:relative;       /* anchor for absolutely-positioned notes */
}}
.paper-text br+br{{display:block;margin-top:.6em}}

/* ── highlights ── */
@keyframes hl-flash{{
  0%   {{filter:brightness(.6) saturate(1.5)}}
  100% {{filter:none}}
}}
.hl{{
  padding:1px 0;cursor:pointer;border-radius:0;
  transition:filter .15s,opacity .2s;
}}
.hl:hover{{filter:brightness(.82)}}
.hl.dimmed{{opacity:.15;pointer-events:none}}
.hl.active{{
  outline:2px solid var(--ring,#000);
  outline-offset:1px;
  animation:hl-flash .3s ease-out;
}}

/* mobile tap indicator */
.hl-tag{{
  display:none;
  font-size:8px;vertical-align:super;
  cursor:pointer;margin-left:1px;
  font-family:'Inter',sans-serif;
}}

/* ── margin notes — WIRED story-row aesthetic ── */
aside.note{{
  position:absolute;
  right:-264px;          /* sits in the paper's right padding zone */
  width:240px;
  background:var(--canvas);
  border:1px solid var(--hairline);
  border-left:3px solid var(--nc,#000);
  border-radius:0;               /* WIRED: square corners */
  padding:10px 12px;
  font-family:'Inter',sans-serif;
  font-size:11px;line-height:1.5;
  cursor:pointer;
  transition:border-color .15s,opacity .25s;
}}
aside.note:hover{{border-color:var(--ink-soft);border-left-color:var(--nc,#000)}}
aside.note.hidden{{display:none!important}}

/* focus mode */
.paper-text.has-active aside.note:not(.active){{opacity:.28}}
aside.note.active{{
  opacity:1!important;
  border:2px solid var(--ink)!important;
  border-left:4px solid var(--nc,#000)!important;
}}

.note-header{{
  display:flex;align-items:baseline;gap:6px;
  margin-bottom:6px;
  padding-bottom:5px;
  border-bottom:1px solid var(--hairline);
}}
.note-badge{{
  font-weight:700;font-size:10px;
  letter-spacing:0.4px;text-transform:uppercase;
}}
.note-tags{{
  color:var(--body-col);font-size:10px;flex:1;text-align:right;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}}
.note-issue{{
  color:var(--ink-soft);margin-bottom:5px;font-size:11.5px;line-height:1.5;
}}
.note-fix{{
  font-family:'Lora',serif;
  font-size:10.5px;font-style:italic;
  color:var(--body-col);line-height:1.5;
}}

/* ── mobile ── */
@media(max-width:860px){{
  .masthead{{padding:0 12px;overflow-x:auto}}
  .masthead-logo{{font-size:16px;padding:10px 14px 10px 0}}
  .masthead-summary{{display:none}}
  .nav-pos{{display:none}}
  .paper{{padding:28px 18px 60px;padding-right:18px;border-left:none;border-right:none}}
  .paper-eyebrow{{font-size:10px}}

  .hl-tag{{display:inline}}

  aside.note{{
    position:static;        /* back to flow on mobile */
    width:calc(100% - 12px);
    margin:0 0 0 8px;
    padding:0;
    border:none;
    border-left:3px solid var(--nc,#000);
    border-radius:0;
    max-height:0;overflow:hidden;
    transition:max-height .3s ease,padding .25s,opacity .2s;
    opacity:0;
  }}
  aside.note.open{{
    max-height:240px;padding:10px 12px;
    border-top:1px solid var(--hairline);
    border-bottom:1px solid var(--hairline);
    border-right:1px solid var(--hairline);
    opacity:1;margin-bottom:12px;
  }}
  .paper-text.has-active aside.note:not(.active){{opacity:1}}
}}
</style>
</head>
<body>

<!-- single masthead: logo | verdict | filters | summary | nav -->
<header class="masthead">
  <a class="masthead-logo" href="#">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 560" width="26" height="26" aria-hidden="true">
      <rect x="80"  y="80"  width="400" height="320" fill="#E8252A"/>
      <rect x="0"   y="280" width="80"  height="80"  fill="#E8252A"/>
      <rect x="480" y="280" width="80"  height="80"  fill="#E8252A"/>
      <rect x="160" y="160" width="40"  height="80"  fill="#ffffff"/>
      <rect x="320" y="160" width="40"  height="80"  fill="#ffffff"/>
      <rect x="200" y="400" width="40"  height="80"  fill="#E8252A"/>
      <rect x="320" y="400" width="40"  height="80"  fill="#E8252A"/>
      <rect x="330" y="290" width="20"  height="20"  fill="#1A1A1A"/>
    </svg>
    <span class="brand-red">red</span>ink
  </a>
  <div class="masthead-meta">
    <span class="vbadge {vcls}">{vstatus}</span>
    <button class="sev-btn on" data-f="all">ALL <span style="font-weight:400;opacity:.6">{n_all}</span></button>
    <button class="sev-btn" data-f="critical" style="color:var(--red)">● {critical} critical</button>
    <button class="sev-btn" data-f="major" style="color:var(--amber)">! {major} major</button>
    <button class="sev-btn" data-f="minor" style="color:var(--blue)">· {minor} minor</button>
    <span class="masthead-summary">{summary}</span>
  </div>
  <div class="nav">
    <span class="nav-pos" id="nav-pos">—</span>
    <button class="nav-btn" id="prev-btn" disabled>&#8592;</button>
    <button class="nav-btn" id="next-btn" disabled>&#8594;</button>
  </div>
</header>

<div class="outer">
  <div class="paper">
    <div class="paper-eyebrow">{title}</div>
    {header}
    <div class="paper-text" id="text">{body}</div>
  </div>
</div>

<script>
const meta = {meta_json};
const isMobile = () => window.innerWidth <= 860;

const textEl  = document.getElementById('text');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const navPos  = document.getElementById('nav-pos');

// ordered list of visible highlights (updates after filter)
let hlList = [];
let curIdx = -1;

function visibleHls() {{
  return [...document.querySelectorAll('.hl:not(.dimmed)')];
}}

function updateNav() {{
  hlList = visibleHls();
  prevBtn.disabled = curIdx <= 0 || hlList.length === 0;
  nextBtn.disabled = curIdx >= hlList.length - 1 || hlList.length === 0;
  navPos.textContent = hlList.length > 0 && curIdx >= 0
    ? `${{curIdx + 1}}/${{hlList.length}}`
    : hlList.length > 0 ? `—/${{hlList.length}}` : '—';
}}

function deactivateAll() {{
  document.querySelectorAll('aside.note.active').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.hl.active').forEach(h => h.classList.remove('active'));
  document.querySelectorAll('aside.note.open').forEach(n => n.classList.remove('open'));
  textEl.classList.remove('has-active');
}}

function activateId(id, scrollNote) {{
  deactivateAll();
  const hl   = document.querySelector(`.hl[data-id="${{id}}"]`);
  const note = document.querySelector(`aside[data-id="${{id}}"]`);
  if (!hl || !note) return;

  // flash + ring
  const ringCol = note.style.getPropertyValue('--nc') || '#888';
  hl.style.setProperty('--ring', ringCol);
  hl.classList.add('active');
  note.classList.add('active');
  textEl.classList.add('has-active');

  if (isMobile()) {{
    note.classList.add('open');
    hl.scrollIntoView({{behavior:'smooth', block:'center'}});
  }} else {{
    hl.scrollIntoView({{behavior:'smooth', block:'center'}});
    if (scrollNote) note.scrollIntoView({{behavior:'smooth', block:'nearest'}});
  }}

  hlList = visibleHls();
  curIdx = hlList.findIndex(h => h.dataset.id === String(id));
  updateNav();
}}

// highlight clicks
document.querySelectorAll('.hl').forEach(hl => {{
  hl.addEventListener('click', e => {{
    e.stopPropagation();
    const id = hl.dataset.id;
    const alreadyActive = hl.classList.contains('active');
    if (alreadyActive) {{ deactivateAll(); curIdx = -1; updateNav(); return; }}
    activateId(id, false);
  }});
}});

// mobile tap-indicator clicks
document.querySelectorAll('.hl-tag').forEach(tag => {{
  tag.addEventListener('click', e => {{
    e.stopPropagation();
    activateId(tag.dataset.id, false);
  }});
}});

// note clicks (desktop → scroll to highlight; mobile handled by hl click above)
document.querySelectorAll('aside.note').forEach(note => {{
  note.addEventListener('click', e => {{
    e.stopPropagation();
    if (!isMobile()) {{
      const alreadyActive = note.classList.contains('active');
      if (alreadyActive) {{ deactivateAll(); curIdx = -1; updateNav(); return; }}
      activateId(note.dataset.id, false);
    }}
  }});
}});

// click outside → deactivate
document.addEventListener('click', () => {{
  deactivateAll();
  curIdx = -1;
  updateNav();
}});

// ← → navigation
prevBtn.addEventListener('click', e => {{
  e.stopPropagation();
  hlList = visibleHls();
  if (curIdx > 0) {{ curIdx--; activateId(hlList[curIdx].dataset.id, true); }}
}});
nextBtn.addEventListener('click', e => {{
  e.stopPropagation();
  hlList = visibleHls();
  if (curIdx < hlList.length - 1) {{ curIdx++; activateId(hlList[curIdx].dataset.id, true); }}
  else if (curIdx === -1 && hlList.length) {{ curIdx = 0; activateId(hlList[0].dataset.id, true); }}
}});

// keyboard
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
    e.preventDefault();
    nextBtn.click();
  }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
    e.preventDefault();
    prevBtn.click();
  }} else if (e.key === 'Escape') {{
    deactivateAll(); curIdx = -1; updateNav();
  }}
}});

// filters (severity buttons in masthead)
let activeFilter = 'all';
document.querySelectorAll('.sev-btn').forEach(btn => {{
  btn.addEventListener('click', e => {{
    e.stopPropagation();
    const f = btn.dataset.f;
    // clicking active filter → reset to all
    if (activeFilter === f && f !== 'all') {{
      activeFilter = 'all';
    }} else {{
      activeFilter = f;
    }}
    document.querySelectorAll('.sev-btn').forEach(b => b.classList.remove('on'));
    document.querySelector(`.sev-btn[data-f="${{activeFilter}}"]`).classList.add('on');

    document.querySelectorAll('aside.note').forEach(n => {{
      n.classList.toggle('hidden', activeFilter !== 'all' && n.dataset.sev !== activeFilter);
    }});
    document.querySelectorAll('.hl').forEach(h => {{
      const sev = [...h.classList].find(c => c.startsWith('hl-'))?.replace('hl-','');
      h.classList.toggle('dimmed', activeFilter !== 'all' && sev !== activeFilter);
    }});
    document.querySelectorAll('.hl-tag').forEach(t => {{
      const hl = document.querySelector(`.hl[data-id="${{t.dataset.id}}"]`);
      t.style.display = hl && !hl.classList.contains('dimmed') ? 'inline' : 'none';
    }});
    deactivateAll(); curIdx = -1;
    updateNav();
    repositionNotes();
  }});
}});

// ── note positioning (absolute, aligned with highlights) ──────────────────
function repositionNotes() {{
  if (isMobile()) {{
    // clear any inline top set by previous desktop positioning
    document.querySelectorAll('aside.note').forEach(n => n.style.top = '');
    return;
  }}
  const notes = [...textEl.querySelectorAll('aside.note:not(.hidden)')];
  let lastBottom = 0;
  const GAP = 8;
  notes.forEach(note => {{
    const hl = textEl.querySelector(`.hl[data-id="${{note.dataset.id}}"]`);
    const idealTop = hl ? hl.offsetTop - 4 : lastBottom;
    const top = Math.max(idealTop, lastBottom + GAP);
    note.style.top = top + 'px';
    lastBottom = top + note.offsetHeight;
  }});
}}

// init
updateNav();
// wait for fonts then position notes (fonts change element heights)
if (document.fonts && document.fonts.ready) {{
  document.fonts.ready.then(() => {{
    repositionNotes();
    // typeset math after notes are placed; reposition once MathJax finishes
    if (window.MathJax && MathJax.typesetPromise) {{
      MathJax.typesetPromise([textEl]).then(repositionNotes);
    }}
  }});
}} else {{
  window.addEventListener('load', () => {{
    repositionNotes();
    if (window.MathJax && MathJax.typesetPromise) {{
      MathJax.typesetPromise([textEl]).then(repositionNotes);
    }}
  }});
}}
window.addEventListener('resize', repositionNotes);
</script>
</body>
</html>
"""
