#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
from pathlib import Path

def compress_folder(source_dir, output_path=None, archive_format='zip'):
    source_dir = Path(source_dir).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"源文件夹不存在或不是目录: {source_dir}")

    if output_path is None:
        base_name = str(source_dir.parent / source_dir.name)
    else:
        output_path = Path(output_path)
        if output_path.suffix in ('.zip', '.tar', '.gz', '.tgz'):
            stem = output_path.stem
            if archive_format == 'gztar' and output_path.suffix == '.gz':
                stem = output_path.name.replace('.tar.gz', '').replace('.tgz', '')
            base_name = str(output_path.parent / stem)
        else:
            base_name = str(output_path)

    fmt_map = {
        'zip': 'zip',
        'tar': 'tar',
        'gztar': 'gztar',
        'tgz': 'gztar'
    }
    fmt = fmt_map.get(archive_format.lower(), 'zip')

    # 修复：提取所有支持的格式名
    supported_formats = [f for f, d in shutil.get_archive_formats()]
    if fmt not in supported_formats:
        raise ValueError(f"不支持的压缩格式: {archive_format}，可用格式: {supported_formats}")

    print(f"正在压缩: {source_dir}")
    archive_path = shutil.make_archive(base_name, fmt, root_dir=source_dir.parent, base_dir=source_dir.name)
    print(f"压缩完成: {archive_path}")
    return archive_path

def main():
    parser = argparse.ArgumentParser(description='将指定文件夹压缩成压缩包')
    parser.add_argument('source', help='要压缩的源文件夹路径')
    parser.add_argument('output', nargs='?', help='输出压缩包路径（可选）')
    parser.add_argument('--format', '-f', default='zip', choices=['zip', 'tar', 'gztar', 'tgz'],
                        help='压缩格式：zip, tar, gztar (tar.gz), tgz，默认 zip')
    args = parser.parse_args()

    try:
        compress_folder(args.source, args.output, args.format)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()