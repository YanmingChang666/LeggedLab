#!/usr/bin/env python3
"""
将meshes目录下所有文件名中的连字符 '-' 替换为下划线 '_'
用法: python3 rename_meshes.py <meshes目录路径>
     python3 rename_meshes.py  (不传参数则使用脚本所在目录)
"""

import os
import sys


def rename_hyphens_to_underscores(directory: str, dry_run: bool = False):
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        print(f"[错误] 目录不存在: {directory}")
        sys.exit(1)

    print(f"目标目录: {directory}")
    if dry_run:
        print("[预览模式] 不会实际重命名，仅显示将要执行的操作\n")

    renamed = 0
    skipped = 0

    for filename in sorted(os.listdir(directory)):
        if '-' not in filename:
            skipped += 1
            continue

        new_filename = filename.replace('-', '_')
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_filename)

        if os.path.exists(dst):
            print(f"[跳过] 目标已存在: {new_filename}")
            skipped += 1
            continue

        print(f"  {filename}  ->  {new_filename}")
        if not dry_run:
            os.rename(src, dst)
        renamed += 1

    print(f"\n完成: 重命名 {renamed} 个文件，跳过 {skipped} 个文件")


if __name__ == "__main__":
    # 支持 --dry-run 参数预览
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    dry_run = '--dry-run' in sys.argv

    if args:
        target_dir = args[0]
    else:
        # 默认使用脚本所在目录
        target_dir = os.path.dirname(os.path.abspath(__file__))

    rename_hyphens_to_underscores(target_dir, dry_run=dry_run)
