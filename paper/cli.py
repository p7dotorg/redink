import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from paper.graph import graph_runner

VERSION = "0.1.0"

HELP = """\
p7review — adversarial pre-submission paper reviewer

Usage:
  p7review <paper.md>           review a paper
  p7review - < paper.md         read from stdin
  paper7 get 2401.12345 | p7review -

Options:
  -h, --help      show this help
  -v, --version   show version

Environment (set in .env or shell):
  OPENROUTER_API_KEY    required for classify/synthesize nodes
  USE_CLAUDE_CLI        true/false  (default: true)
  USE_KIMI_CLI          true/false  (default: false)
  LANGSMITH_TRACING     true/false  (default: false)
"""

SEVERITY_EMOJI = {"critical": "🔴", "major": "🟡", "minor": "🔵"}
SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}
PERSONA_LABEL = {"skeptic": "SKEPTIC", "practitioner": "PRACTITIONER", "academic": "ACADEMIC"}


def format_report(verdict) -> str:
    lines = []
    lines.append("=" * 60)
    status_emoji = {"PASS": "✅", "REVISE": "⚠️", "FAIL": "❌"}
    lines.append(f"  VEREDITO: {status_emoji.get(verdict.status, '')} {verdict.status}")
    lines.append("=" * 60)
    lines.append(f"\n{verdict.summary}\n")
    lines.append(f"Críticos: {verdict.critical_count}  |  Maiores: {verdict.major_count}  |  Menores: {verdict.minor_count}")

    # High-confidence issues (consensus)
    if verdict.high_confidence_issues:
        lines.append(f"\n🎯 CONSENSO (alta confiança):")
        for issue in verdict.high_confidence_issues[:4]:
            lines.append(f"   • {issue}")

    lines.append("")

    # Findings por dimensão + persona
    sorted_findings = sorted(verdict.findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    current_dim = None
    for f in sorted_findings:
        if f.dimension != current_dim:
            current_dim = f.dimension
            lines.append(f"\n── {f.dimension.upper()} " + "─" * (40 - len(f.dimension)))

        emoji = SEVERITY_EMOJI.get(f.severity, "⚪")
        persona = PERSONA_LABEL.get(getattr(f, "persona", ""), "")
        conf = getattr(f, "confidence", 5)
        conf_str = f" [conf:{conf}/10]" if conf != 5 else ""
        lines.append(f"\n{emoji} [{f.severity.upper()}][{persona}]{conf_str} {f.issue}")
        lines.append(f"   Evidência: {f.evidence}")
        lines.append(f"   Sugestão:  {f.suggestion}")

    # Contradiction map
    c_map = getattr(verdict, "contradiction_map", None)
    if c_map and c_map.contradictions:
        lines.append(f"\n\n── CONTRADICTION MAP " + "─" * 38)
        for c in c_map.contradictions[:4]:
            sig = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(c.significance, "")
            lines.append(f"\n{sig} [{c.dimension}] {c.persona_a} vs {c.persona_b}")
            lines.append(f"   A: {c.claim_a}")
            lines.append(f"   B: {c.claim_b}")
        if c_map.most_disputed_dimension:
            lines.append(f"\n⚡ Dimensão mais disputada: {c_map.most_disputed_dimension}")

    # Blind spots
    b_spots = getattr(verdict, "blind_spots", None)
    if b_spots and b_spots.topics_not_covered:
        lines.append(f"\n\n── BLIND SPOTS (nenhum revisor mencionou) " + "─" * 18)
        for topic in b_spots.topics_not_covered[:3]:
            lines.append(f"   ⬡ {topic}")
        if b_spots.highest_priority:
            lines.append(f"\n   ⚠️  Mais crítico: {b_spots.highest_priority}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(HELP)
        sys.exit(0)

    if args[0] in ("-v", "--version"):
        print(f"p7review {VERSION}")
        sys.exit(0)

    target = args[0]
    github_url, paper_text = None, None
    if target == "-":
        paper_text = sys.stdin.read()
        display_name = "stdin"
    elif target.startswith("https://"):
        github_url, display_name = target, target.rsplit("/", 1)[-1]
    else:
        path = Path(target)
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        paper_text = path.read_text(encoding="utf-8")
        display_name = path.name

    input_state = {"github_url": github_url} if github_url else {"paper": paper_text}
    print(f"\np7review: {display_name}")
    print("Classificando paper...")

    result = graph_runner.invoke({
        **input_state, "findings": [],
        "classification": None, "contradiction_map": None,
        "blind_spots": None, "verdict": None,
    })

    clf = result["classification"]
    print(f"Área: {clf.area} | Tipo: {clf.paper_type}")
    print(f"Dimensões acionadas: {', '.join(clf.dimensions)}")
    print(f"Citações encontradas: {len(clf.citations)}")
    print("\nRevisores rodando em paralelo...\n")

    report = format_report(result["verdict"])
    print(report)

    # Salva relatório (só se entrada foi arquivo ou URL, não stdin)
    if target != "-":
        if target.startswith("https://"):
            out_path = Path(f"{display_name}.review.md")
        else:
            out_path = Path(target).with_suffix(".review.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# p7review: {display_name}\n\n")
            f.write(report)
        print(f"\nSalvo em: {out_path}")


if __name__ == "__main__":
    main()
