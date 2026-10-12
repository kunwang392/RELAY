"""Teacher-only formatting of signed rubric context."""

import re


def strip_rubric_metadata(context: str, keep_polarity: bool = True) -> str:
    """Keep criterion bodies, removing weights and criterion headings.

    Polarity labels are retained by default, or omitted for plain criteria.
    Only numbered signed rubrics are changed. Other context and continuation
    lines are preserved, including colons inside the actual criterion body.
    """
    lines = []
    for line in context.splitlines(keepends=True):
        match = re.match(
            r"^(\s*\d+\.\s*)\[(POSITIVE REQUIREMENT|NEGATIVE PENALTY)"
            r",\s*weight=[+-]?\d+\]\s*([^\r\n]*)(\r?\n)?$",
            line,
        )
        if match is None:
            lines.append(line)
            continue
        number, polarity, body, ending = match.groups()
        body = re.sub(
            r"^(?:[^\r\n]*?:\s*)?[A-Za-z]+\s+Criteria:\s*",
            "",
            body,
            count=1,
            flags=re.IGNORECASE,
        )
        label = f"[{polarity}] " if keep_polarity else ""
        lines.append(f"{number}{label}{body}{ending or ''}")
    return "".join(lines)
