#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本功能：读取parquet文件，并根据question_type进行拆分为多个parquet文件
"""

import os
import argparse
import pandas as pd
from typing import List, Dict, Any, Set
from collections import defaultdict

def split_dataset_by_question_type(
    input_file: str,
    output_dir: str = None,
    prefix: str = None,
    question_type_column: str = "question_type"
) -> Dict[str, int]:
    """
    读取parquet文件，并根据question_type拆分为多个parquet文件
    
    Args:
        input_file: 输入的parquet文件路径
        output_dir: 输出文件夹路径，默认与输入文件同目录
        prefix: 输出文件名前缀，默认使用输入文件名
        question_type_column: 问题类型列名，默认为"question_type"
        
    Returns:
        Dict[str, int]: 每种问题类型的样本数量统计
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取parquet文件
    df = pd.read_parquet(input_file)
    total_samples = len(df)
    print(f"总共读取了 {total_samples} 个样本")
    
    # 设置输出文件夹
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件前缀
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(input_file))[0]
    
    # 检查是否存在question_type列
    if question_type_column not in df.columns:
        raise ValueError(f"警告: 数据集中不存在 '{question_type_column}' 列")

    # 获取所有的问题类型
    question_types = df[question_type_column].unique()
    print(f"发现 {len(question_types)} 种问题类型: {', '.join(question_types)}")
    
    # 按问题类型拆分并保存
    type_counts = {}
    for qtype in question_types:
        # 提取该类型的数据
        type_df = df[df[question_type_column] == qtype]
        type_count = len(type_df)
        type_counts[qtype] = type_count
        
        # 构建输出文件路径
        output_file = os.path.join(output_dir, f"{prefix}_{qtype}.parquet")
        
        # 保存到新文件
        type_df.to_parquet(output_file)
        print(f"保存了 {type_count} 个 '{qtype}' 类型样本到: {output_file}")
    
    for qtype in question_types:
        # 提取该类型的数据
        type_df = df[df[question_type_column] != qtype]
        type_count = len(type_df)
        # type_counts[qtype] = type_count
        
        # 构建输出文件路径
        output_file = os.path.join(output_dir, f"{prefix}_wo_{qtype}.parquet")
        
        # 保存到新文件
        type_df.to_parquet(output_file)
        print(f"保存了 {type_count} 个 'wo {qtype}' 类型样本到: {output_file}")
    
    return type_counts

def main():
    parser = argparse.ArgumentParser(description='将parquet文件按question_type拆分为多个文件')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入的parquet文件路径')
    parser.add_argument('--output_dir', '-o', type=str, default='./', help='输出文件夹路径，默认与输入文件同目录')
    parser.add_argument('--prefix', '-p', type=str, default=None, help='输出文件名前缀，默认使用输入文件名')
    parser.add_argument('--column', '-c', type=str, default="question_type", help='问题类型列名，默认为"question_type"')
    
    args = parser.parse_args()
    

    split_dataset_by_question_type(
        input_file=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix,
        question_type_column=args.column
    )

if __name__ == "__main__":
    main()
