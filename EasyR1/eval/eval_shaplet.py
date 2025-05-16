import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
import re
import io
import base64
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2ForCausalLM
from qwen_vl_utils import process_vision_info
ONE_SHOT = False
FEW_SHOT = False

def extract_boxed_content(text: str) -> Union[str, None]:
    """从文本中提取最后一个 \boxed{} 中的内容"""
    pattern = r"\\boxed\{(.*?)\}"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        # 获取最后一个匹配项
        last_match = matches[-1]
        return last_match.group(1)
    return None

def extract_shaplet_values(text: str) -> Union[List[str], None]:
    """从 \boxed{A,B,C,D,E} 标签中提取多个选项值"""
    content = extract_boxed_content(text)
    if not content:
        return None
    try:
        # 将内容按逗号分割
        values = content.split(',')
        if len(values) == 0:
            return None
        
        # 清理每个值（去除空白）
        cleaned_values = [value.strip() for value in values]
        
        return cleaned_values
    except:
        return None

def calculate_accuracy(pred_values: List[str], gt_values: List[str]) -> Dict[str, bool]:
    """
    计算每个选项的准确性（精确匹配）
    
    Args:
        pred_values: 预测的值列表
        gt_values: 真实的值列表
        
    Returns:
        accuracy_dict: 包含每个真实标签预测是否正确的字典
    """
    if pred_values is None or gt_values is None:
        return {}
    
    # 创建结果字典
    result = {'all_correct': False}
    
    # 如果预测值和真实值长度不同，整体准确性为False
    if len(pred_values) != len(gt_values):
        result['all_correct'] = False
        # 为每个真实值添加结果（全部标记为False）
        for i, gt_val in enumerate(gt_values):
            result[gt_val] = False
        return result
    
    # 检查每个值是否完全匹配
    all_correct = True
    for i, (pred, gt) in enumerate(zip(pred_values, gt_values)):
        correct = (pred == gt)
        result[gt] = correct
        if not correct:
            all_correct = False
    
    result['all_correct'] = all_correct
    return result

def load_model(model_path: str) -> Tuple[Any, Any]:
    """
    加载模型和处理器
    
    Args:
        model_path: 模型路径
        model_type: 模型类型，可选值：
            - "auto": 自动检测模型类型
            - "qwen-vl": Qwen-VL模型
            - "qwen": Qwen模型
            - "other": 其他类型模型
    """
    
    # 读取config.json获取模型架构
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_type = config.get("model_type", None)
    else:
        model_type = None
    print(f"Loading model from {model_path} with type {model_type}")
    
    if model_type == "qwen2_5_vl":
        # 设置图像处理的参数
        min_pixels = 256 * 28 * 28  # 最小像素数
        max_pixels = 1280 * 28 * 28  # 最大像素数
        
        # 加载处理器和模型
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        # 根据config.json中的架构类型加载对应的模型类
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
        )
    return model, model_type, processor


def image_to_base64(image_bytes: bytes) -> str:
    """
    将图像字节数据转换为base64格式
    
    Args:
        image_bytes: 图像的字节数据
    
    Returns:
        base64编码的图像数据字符串
    """
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

def process_input(model_type: str, processor: Any, prompt: str, image_bytes: bytes = None) -> Dict:
    """
    处理输入，根据模型类型返回适当的输入格式
    
    Args:
        model_type: 模型类型
        processor: 处理器（可能是tokenizer或processor）
        prompt: 提示词
        image_bytes: 图像字节数据（如果有的话）
    """
    print(model_type)
    if model_type == "qwen2_5_vl" and image_bytes is not None:
        # 创建输出目录
        # os.makedirs("output_images", exist_ok=True)
        
        # # 将图像字节数据转换为PIL Image并保存
        # image = Image.open(io.BytesIO(image_bytes))
        
        # # 获取当前时间戳作为文件名
        # import time
        # timestamp = int(time.time())
        # image_path = f"output_images/image_{timestamp}.png"
        
        # # 保存图像
        # image.save(image_path)
        # print(f"图像已保存至: {image_path}")
        
        # 将图像转换为base64格式
        image_base64 = image_to_base64(image_bytes)
        
        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_base64},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        # 使用处理器处理输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        # inputs = processor(text=text, return_tensors="pt")
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs
    elif model_type == "qwen2_5_vl" and image_bytes is None:
        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    # {"type": "image", "image": image_base64},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        # 使用处理器处理输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = processor(text=text, return_tensors="pt")
        inputs = processor(
            text=[text],
            # images=image_inputs,
            # videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs
    else:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return processor([text], return_tensors="pt")


def main(TYPE: str = "image"):
    parser = argparse.ArgumentParser(description='Evaluate model on shaplet detection task')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--data_path', type=str, default="../data/shaplet_eval.parquet",
                      help='Path to the evaluation data')
    parser.add_argument('--output_path', type=str, default="./results_shaplet.json",
                      help='Path to save the results')
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model, model_type, processor = load_model(args.model_path)
    
    # 加载评估数据
    print(f"Loading evaluation data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    # 初始化统计计数器
    total_samples = 0
    all_correct_count = 0
    label_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}  # 每个标签出现的总次数
    correct_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}  # 每个标签预测正确的次数
    results = []
    
    # 遍历每个样本进行评估
    for idx, row in df.iterrows():
        print(f"处理样本 {idx + 1}/{len(df)}...")
        
        # 从row中获取图像数据
        try:
            image_data = row['images'][0]  # 获取第一个（也是唯一的）图像数据
            image_bytes = image_data['bytes']  # 获取图像字节数据
        except:
            image_bytes = None
        
        # 构建提示词
        prompt = row['problem']
        
        # 生成回答
        inputs = process_input(model_type, processor, prompt, image_bytes)
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=4096, repetition_penalty=1.1)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print("模型输出结果:", response)

        # 提取预测的shaplet值
        pred_values = extract_shaplet_values(response[0])
        
        # 从solution中提取真实的shaplet值
        gt_values = extract_shaplet_values(row['solution'])

        print("预测答案: ",pred_values)
        print("真实答案: ",gt_values)
        
        # 计算准确性
        accuracy_dict = calculate_accuracy(pred_values, gt_values)
        print("准确性: ",accuracy_dict)
        
        # 累积统计量
        total_samples += 1
        if accuracy_dict.get('all_correct', False):
            all_correct_count += 1
        
        # 更新每个标签的统计
        if gt_values:
            for label in gt_values:
                if label in label_counts:
                    label_counts[label] += 1
                    if accuracy_dict.get(label, False):
                        correct_counts[label] += 1
        
        # 保存结果
        results.append({
            'sample_id': idx,
            'prediction': pred_values,
            'ground_truth': gt_values,
            'accuracy': accuracy_dict
        })
        
        # 打印每个样本的结果
        print(f"样本 {idx + 1} - 整体正确: {accuracy_dict.get('all_correct', False)}")
        if gt_values:
            for label in gt_values:
                print(f"  标签 {label}: {accuracy_dict.get(label, False)}")
    
    # 计算整体准确率
    overall_accuracy = all_correct_count / total_samples if total_samples > 0 else 0
    
    # 计算每个标签的准确率
    label_accuracies = {}
    for label in ['A', 'B', 'C', 'D', 'E']:
        if label_counts[label] > 0:
            label_accuracies[label] = correct_counts[label] / label_counts[label]
        else:
            label_accuracies[label] = 0
    
    print("\n整体结果:")
    print(f"整体准确率: {overall_accuracy:.4f}")
    print(f"总样本数: {total_samples}")
    print(f"整体正确数: {all_correct_count}")
    
    print("\n各标签准确率:")
    for label in ['A', 'B', 'C', 'D', 'E']:
        print(f"标签 {label}:")
        print(f"  总计: {label_counts[label]}")
        print(f"  正确: {correct_counts[label]}")
        print(f"  准确率: {label_accuracies[label]:.4f}")
    
    # 保存详细结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump({
            'overall': {
                'accuracy': overall_accuracy,
                'total_samples': total_samples,
                'correct_count': all_correct_count
            },
            'label_stats': {
                label: {
                    'total': label_counts[label],
                    'correct': correct_counts[label],
                    'accuracy': label_accuracies[label]
                } for label in ['A', 'B', 'C', 'D', 'E']
            },
            'samples': results
        }, f, indent=2)
    
    print(f"\n详细结果已保存至 {args.output_path}")

if __name__ == "__main__":
    main(TYPE="text") 