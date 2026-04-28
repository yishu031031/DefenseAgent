"""Reference implementation referenced from the tabular-report skill body."""


def render_table(rows: list[dict], columns: list[str]) -> str:
    """Render `rows` as a GitHub-flavored Markdown table in `columns` order."""
    if not rows:
        return "(no rows)"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines: list[str] = []
    for row in rows:
        cells: list[str] = []
        for col in columns:
            cells.append(str(row.get(col, "")))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + body_lines)
