import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Desktop\mycode_raw\rl\multi_obj_gspo.py', 'rb') as f:
    lines = f.readlines()

# 替换第158行 - 减少外循环次数
new_158 = '#         iterations=100,  # 减少外循环次数\n'
lines[157] = new_158.encode('utf-8')

# 替换第335行 - idx注释
new_335 = '        self.buffer = []  # (reward, idx, token_ids, smiles) idx用于打破reward相等时的比较\n'
lines[334] = new_335.encode('utf-8')

# 替换第337行 - counter注释  
new_337 = '        self._counter = 0  # 唯一计数器，用于避免tuple比较时比token_ids\n'
lines[336] = new_337.encode('utf-8')

# 替换第607行 - 打印结果
new_607 = '    # 5. 打印结果\n'
lines[606] = new_607.encode('utf-8')

with open(r'd:\Desktop\mycode_raw\rl\multi_obj_gspo.py', 'wb') as f:
    f.writelines(lines)

print('All Done!')