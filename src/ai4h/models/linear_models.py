from pathlib import Path


def test():
    project_root = Path(__file__).resolve().parents[3]
    relpath = Path(__file__).resolve().relative_to(project_root)
    print(f"Hello from `{relpath}`!")
