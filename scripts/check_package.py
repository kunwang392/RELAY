"""Check source syntax, local imports, and accidental data/credential inclusion."""
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOTS = {"roll": ROOT / "roll", "mcore_adapter": ROOT / "mcore_adapter/src/mcore_adapter"}
FORBIDDEN_SUFFIXES = {".parquet", ".arrow", ".jsonl", ".safetensors", ".pt", ".pth", ".ckpt", ".log", ".pyc",
                      ".pdf", ".png", ".jpg", ".svg", ".ipynb"}


def module_exists(name):
    first, *parts = name.split('.')
    if first not in MODULE_ROOTS:
        return True
    path = MODULE_ROOTS[first].joinpath(*parts)
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file()


def inspect():
    findings = []
    count = 0
    patterns = {
        'private_home_path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
        'internal_domain': re.compile(r'(?:alibaba-inc[.]com|aliyun-inc[.]com|sre[_]agent[_]train)'),
        'private_network_address': re.compile(r'(?<!\d)(?:10|11|33|172|192)[.]\d+[.]\d+[.]\d+'),
        'token_literal': re.compile(r'''(?i)(?:api[_-]?key|access[_-]?key|password|authorization)\s*[:=]\s*["'](?:sk-|AKIA|Bearer |t-)[A-Za-z0-9_+/-]{8,}'''),
        'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    }
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(ROOT))
        if any(p in {'.git', '.venv', 'output', '.pytest_cache', '__pycache__'} for p in path.relative_to(ROOT).parts):
            findings.append({'file': relative, 'kind': 'runtime residue'})
            continue
        if path.suffix in FORBIDDEN_SUFFIXES:
            findings.append({'file': relative, 'kind': 'data, binary, or log'})
            continue
        text = path.read_text()
        for line, content in enumerate(text.splitlines(), 1):
            for kind, pattern in patterns.items():
                if pattern.search(content):
                    findings.append({'file': relative, 'line': line, 'kind': kind})
        if path.suffix != '.py':
            continue
        count += 1
        tree = ast.parse(text, filename=relative)
        compile(tree, relative, 'exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or '']
            else:
                continue
            for name in names:
                if name.split('.')[0] in {'matplotlib', 'seaborn', 'plotly'}:
                    findings.append({'file': relative, 'line': node.lineno, 'kind': 'plotting dependency'})
                if name.split('.')[0] in MODULE_ROOTS and not module_exists(name):
                    findings.append({'file': relative, 'line': node.lineno, 'kind': 'missing local module', 'module': name})
    return {'python_files': count, 'findings': findings}


if __name__ == '__main__':
    result = inspect()
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['findings']))
