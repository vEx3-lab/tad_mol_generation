"""配置加载工具。

负责读取 YAML 配置，并提前创建配置中声明的输出目录。这里不做训练参数
解释，避免配置加载阶段混入模型或实验逻辑。
"""

import os

import yaml

from project_paths import resolve_project_path


def load_config(config_file):
    """读取 YAML 配置，并创建 `paths` 下涉及的目录。"""
    config_file = resolve_project_path(config_file)
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for _, path in config.get("paths", {}).items():
        resolved_path = resolve_project_path(path)
        dir_path = resolved_path if os.path.isdir(resolved_path) else resolved_path.parent
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    return config
