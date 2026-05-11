"""生成项目文件盘点摘要。

用途：
    python scripts/project_inventory.py

这个脚本只读取目录和文件元信息，不修改任何项目文件。它用于在整理项目前
快速确认入口脚本、配置文件、日志文件和大型实验产物的位置。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORE_DIRS = {".git", ".idea", "__pycache__"}
ENTRY_PATTERNS = ("if __name__", "def main(", "ArgumentParser", "argparse")


def iter_files():
    """遍历项目文件，跳过常见缓存目录。"""
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        yield path


def relative(path: Path) -> str:
    """返回便于阅读的项目相对路径。"""
    return path.relative_to(PROJECT_ROOT).as_posix()


def has_entry_point(path: Path) -> bool:
    """判断 Python 文件是否像可直接运行的入口脚本。"""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(pattern in text for pattern in ENTRY_PATTERNS)


def main() -> None:
    files = list(iter_files())
    extensions = Counter(path.suffix or "<no_ext>" for path in files)

    print("Project root:", PROJECT_ROOT)
    print("Total files:", len(files))
    print()

    print("File types:")
    for ext, count in extensions.most_common(20):
        print(f"  {ext}: {count}")
    print()

    print("Python entry candidates:")
    for path in sorted(p for p in files if p.suffix == ".py" and has_entry_point(p)):
        print("  " + relative(path))
    print()

    print("Large Python files:")
    python_files = sorted(
        (path for path in files if path.suffix == ".py"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    for path in python_files[:20]:
        print(f"  {relative(path)} ({path.stat().st_size} bytes)")
    print()

    print("Recent GSPO logs:")
    log_dir = PROJECT_ROOT / "rl" / "feedback" / "logs"
    if log_dir.exists():
        logs = sorted(
            log_dir.glob("gspo_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in logs[:10]:
            print(f"  {relative(path)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
