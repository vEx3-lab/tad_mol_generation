"""项目路径工具。

本文件只做路径解析与目录创建，不包含训练逻辑。保留原有函数名，
让旧脚本可以继续 `from project_paths import resolve_project_path`。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT


def project_path(*parts):
    """返回项目根目录下的路径。"""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_path(path):
    """把配置文件中的相对路径解析到项目根目录下。

    兼容历史配置里常见的 `../data/...` 写法：项目脚本通常从根目录
    运行，这里的 `..` 实际表示项目根目录的上一层，容易造成混乱；
    因此保留旧行为，将开头的 `..` 去掉后再拼接到项目根目录。
    """
    path = Path(path)
    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == "..":
        path = Path(*path.parts[1:])

    return PROJECT_ROOT / path


def ensure_dir(path):
    """确保目录存在，并返回该目录路径。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path):
    """确保文件路径的父目录存在，并返回原路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
