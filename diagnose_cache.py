#!/usr/bin/env python3
"""诊断 datasets 缓存问题的脚本"""

import os
import sys
import glob
import hashlib

# 模拟 train_flashmtp.py 的参数
train_data_path = './cache/dataset/train/nemotron_400000_train_regen.jsonl'  # 请修改为你的实际路径
max_length = 4096
chat_template = 'qwen'
target_model_path = '/share/wanghanzhen/models/Qwen/Qwen3-8B'  # 请修改为你的实际路径
cache_dir = './cache/train'
build_dataset_num_proc = 8
is_preformatted = True

print("=" * 60)
print("诊断 datasets 缓存问题")
print("=" * 60)

# 计算 cache_key（与代码中相同）
cache_params_string = (
    f"{train_data_path}-"
    f"{max_length}-"
    f"{chat_template}-"
    f"{target_model_path}"
)
cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
print(f"\n1. 计算的 cache_key: {cache_key}")

# 预期的缓存目录
full_cache_dir = os.path.join(cache_dir, "processed_dataset")
print(f"2. 缓存目录: {full_cache_dir}")

# 检查实际存在的文件
existing_files = glob.glob(os.path.join(full_cache_dir, "*.pkl"))
print(f"\n3. 缓存目录中所有 .pkl 文件数量: {len(existing_files)}")

# 按文件名分组
from collections import defaultdict
cache_groups = defaultdict(list)
for f in existing_files:
    basename = os.path.basename(f)
    # 提取 fingerprint 部分（去掉 _0000X_of_0000X.pkl 后缀）
    if '_of_' in basename:
        fingerprint = basename.split('_')[0]
        cache_groups[fingerprint].append(f)

print(f"\n4. 发现 {len(cache_groups)} 个不同的缓存 fingerprint:")
for fp, files in sorted(cache_groups.items()):
    print(f"   - {fp}: {len(files)} 个分片文件")

print(f"\n5. 当前计算出的 fingerprint: {cache_key}")
if cache_key in cache_groups:
    print("   ✓ 匹配到已有缓存！")
    print(f"   文件数量: {len(cache_groups[cache_key])}")
else:
    print("   ✗ 没有匹配的缓存，需要重新生成")
    print("\n   可能的原因:")
    print("   - 数据路径改变")
    print("   - max_length 参数改变")
    print("   - chat_template 改变")
    print("   - target_model_path 改变")

# 检查 arrow 缓存文件
arrow_files = glob.glob(os.path.join(full_cache_dir, "*.arrow"))
print(f"\n6. Arrow 缓存文件数量: {len(arrow_files)}")
for f in arrow_files:
    print(f"   - {os.path.basename(f)}")

print("\n" + "=" * 60)
print("解决方案:")
print("=" * 60)
print("""
如果 fingerprint 不匹配，有以下选择:

1. 重新生成缓存（当前行为）
   - 等待 map 完成

2. 使用已有的缓存文件
   - 修改 cache_key 为已有 fingerprint
   - 或直接重命名缓存文件

3. 检查参数是否与之前一致
   - 数据路径
   - max_length
   - chat_template
   - target_model_path
""")
