import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

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


def format_report(verdict) -> str:
    lines = []
    lines.append("=" * 60)
    status_emoji = {"PASS": "✅", "REVISE": "⚠️", "FAIL": "❌"}
    lines.append(f"  VEREDITO: {status_emoji.get(verdict.status, '')} {verdict.status}")
    lines.append("=" * 60)
    lines.append(f"\n{verdict.summary}\n")
    lines.append(f"Críticos: {verdict.critical_count}  |  Maiores: {verdict.major_count}  |  Menores: {verdict.minor_count}")
    lines.append("")

    sorted_findings = sorted(verdict.findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    current_dim = None
    for f in sorted_findings:
        if f.dimension != current_dim:
            current_dim = f.dimension
            lines.append(f"\n── {f.dimension.upper()} " + "─" * (40 - len(f.dimension)))

        emoji = SEVERITY_EMOJI.get(f.severity, "⚪")
        lines.append(f"\n{emoji} [{f.severity.upper()}] {f.issue}")
        lines.append(f"   Evidência: {f.evidence}")
        lines.append(f"   Sugestão:  {f.suggestion}")

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

    if target == "-":
        paper_text = sys.stdin.read()
        display_name = "stdin"
    else:
        path = Path(target)
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        paper_text = path.read_text(encoding="utf-8")
        display_name = path.name
    print(f"\np7review: {display_name} ({len(paper_text)} chars)")
    print("Classificando paper...")

    result = graph.invoke({"paper": paper_text, "findings": [], "classification": None, "verdict": None})

    clf = result["classification"]
    print(f"Área: {clf.area} | Tipo: {clf.paper_type}")
    print(f"Dimensões acionadas: {', '.join(clf.dimensions)}")
    print(f"Citações encontradas: {len(clf.citations)}")
    print("\nRevisores rodando em paralelo...\n")

    report = format_report(result["verdict"])
    print(report)

    # Salva relatório (só se entrada foi arquivo, não stdin)
    if target != "-":
        out_path = Path(target).with_suffix(".review.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# p7review: {display_name}\n\n")
            f.write(report)
        print(f"\nSalvo em: {out_path}")


if __name__ == "__main__":
    main()
