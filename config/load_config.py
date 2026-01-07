import os
import yaml

def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 只创建目录，不创建文件
    for k, path in config.get("paths", {}).items():
        # 如果路径是文件，取上级目录
        dir_path = path if os.path.isdir(path) else os.path.dirname(path)
        if dir_path:  # 避免空字符串
            os.makedirs(dir_path, exist_ok=True)

    return config
