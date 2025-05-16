# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import numpy as np
from typing import List, Union, Tuple

from mathruler.grader import extract_boxed_content, grade_answer


def extract_period(text: str) -> Union[int, float, None]:
    """Extract period value from \period{} tag"""
    pattern = r"\\period\{([^}]+)\}"
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_recon(text: str) -> Union[List[float], None]:
    """Extract reconstructed series from \recon{} tag"""
    pattern = r"\\recon\{([^}]+)\}"
    match = re.search(pattern, text)
    if match:
        try:
            series_str = match.group(1)
            # 移除所有空白字符并分割
            series = [float(x) for x in series_str.strip('[]').replace(' ', '').split(',')]
            return series
        except (ValueError, SyntaxError):
            return None
    return None


def extract_anomaly_intervals(text: str) -> Union[List[List[int]], None]:
    """Extract anomaly intervals from \boxed{} tag"""
    content = extract_boxed_content(text)
    if not content:
        return None
    try:
        # 评估字符串为Python列表
        intervals = eval(content)
        if not isinstance(intervals, list):
            return None
            
        processed_intervals = []
        for x in intervals:
            if not isinstance(x, list):
                return None
                
            # 处理单点情况
            if len(x) == 1:
                if isinstance(x[0], (int, float)):
                    processed_intervals.append([int(x[0]), int(x[0])])
            # 处理区间情况
            elif len(x) == 2:
                if (isinstance(x[0], (int, float)) and 
                    isinstance(x[1], (int, float)) and 
                    x[0] <= x[1]):
                    processed_intervals.append([int(x[0]), int(x[1])])
            else:
                return None
                
        # 确保区间是有序的
        processed_intervals.sort(key=lambda x: x[0])
        return processed_intervals
    except:
        return None


def ts_format_reward_cot(predict_str: str) -> float:
    """检查预测结果是否包含所有必需的标签和格式"""
    required_patterns = [
        r"</think>",
        r"\\period\{[^}]+\}",
        r"\\recon\{[^}]+\}",
        r"\\boxed\{.*\}"
    ]
    
    score = 0.0
    for pattern in required_patterns:
        if re.search(pattern, predict_str, re.DOTALL):
            score += 0.25
    return score

def ts_format_reward(predict_str: str) -> float:
    """检查预测结果是否包含所有必需的标签和格式"""
    required_patterns = [
        r"</think>",
        r"\\boxed\{.*\}"
    ]
    
    score = 0.0
    for pattern in required_patterns:
        if re.search(pattern, predict_str, re.DOTALL):
            score += 0.5
    return score


def period_reward(predict_str: str, ground_truth_str: str) -> float:
    """计算预测的周期值与真实周期值的相似度"""
    pred_period = extract_period(predict_str)
    gt_period = extract_period(ground_truth_str)
    
    if pred_period is None or gt_period is None:
        return 0.0
    
    # 使用相对误差计算周期的准确性
    relative_error = abs(pred_period - gt_period) / gt_period
    return max(0, 1 - relative_error)


def recon_reward(predict_str: str, ground_truth_str: str) -> float:
    """计算重构序列与真实重构序列的相似度"""
    pred_recon = extract_recon(predict_str)
    gt_recon = extract_recon(ground_truth_str)
    
    if pred_recon is None or gt_recon is None:
        return 0.0
    
    if len(pred_recon) != len(gt_recon):
        return 0.0
    
    # 使用均方根误差(RMSE)计算重构序列的准确性
    rmse = np.sqrt(np.mean((np.array(pred_recon) - np.array(gt_recon)) ** 2))
    max_val = max(abs(max(gt_recon)), abs(min(gt_recon)))
    normalized_rmse = rmse / max_val
    return max(0, 1 - normalized_rmse)


def anomaly_intervals_reward(predict_str: str, ground_truth_str: str) -> float:
    """计算异常区间检测的准确性，采用改进的评分机制，包含对重叠区间的惩罚"""
    pred_intervals = extract_anomaly_intervals(predict_str)
    gt_intervals = extract_anomaly_intervals(ground_truth_str)
    
    if pred_intervals is None or gt_intervals is None:
        return 0.0
    
    # 如果真实值和预测值都是空列表，返回满分
    if not gt_intervals and not pred_intervals:
        return 1.0
    
    # 如果其中一个是空列表而另一个不是，返回0分
    if not gt_intervals or not pred_intervals:
        return 0.0
        
    # 检查预测区间是否有重叠
    overlap_penalty = 0.0
    if len(pred_intervals) > 1:
        for i in range(len(pred_intervals)-1):
            for j in range(i+1, len(pred_intervals)):
                # 检查区间是否重叠
                if pred_intervals[i][1] >= pred_intervals[j][0]:
                    # 计算重叠长度占区间总长度的比例
                    overlap_len = min(pred_intervals[i][1], pred_intervals[j][1]) - pred_intervals[j][0] + 1
                    total_len = max(pred_intervals[i][1], pred_intervals[j][1]) - min(pred_intervals[i][0], pred_intervals[j][0]) + 1
                    overlap_ratio = overlap_len / total_len
                    overlap_penalty += overlap_ratio
    
    # 归一化重叠惩罚值，最大惩罚为0.5
    if overlap_penalty > 0:
        overlap_penalty = min(0.5, overlap_penalty / len(pred_intervals))
    
    # 计算point-adjust F1分数
    # 假设时间序列长度为最大区间结束位置+1
    ts_length = max(
        max([interval[1] for interval in pred_intervals] + [0]),
        max([interval[1] for interval in gt_intervals] + [0])
    ) + 1
    
    # 创建预测和真实的点标签序列
    pred_labels = np.zeros(ts_length, dtype=bool)
    gt_labels = np.zeros(ts_length, dtype=bool)
    
    # 填充预测标签
    for start, end in pred_intervals:
        pred_labels[start:end+1] = True
        
    # 填充真实标签，并扩展区间范围（前后各5个点）
    gt_intervals_after_enlarge = []
    for start, end in gt_intervals:
        start = max(start - 5, 0)
        end = min(end + 5, ts_length - 1)
        gt_labels[start:end+1] = True
        gt_intervals_after_enlarge.append([start, end])

    # 执行point adjustment
    adjusted_pred_labels = np.copy(pred_labels)
    for start, end in gt_intervals_after_enlarge:
        if np.any(pred_labels[start:end+1]):
            adjusted_pred_labels[start:end+1] = True

    # 计算TP、FP、FN
    tp = np.sum(np.logical_and(adjusted_pred_labels, gt_labels))
    fp = np.sum(np.logical_and(adjusted_pred_labels, ~gt_labels))
    fn = np.sum(np.logical_and(~adjusted_pred_labels, gt_labels))
    
    # 计算precision和recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 计算F1
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 最终分数 = F1分数 * (1 - overlap_penalty)
    final_score = f1_score * (1 - overlap_penalty)
    
    return final_score


def extract_choice_answers(text: str) -> Union[List[str], None]:
    """从\boxed{}标签中提取选择题答案"""
    content = extract_boxed_content(text)
    if not content:
        return None
    try:
        # 尝试评估字符串为Python列表
        choices = eval(content)
        if isinstance(choices, list) and all(isinstance(choice, str) for choice in choices):
            return choices
        # 如果是单个字符串且是单个字母，作为单选题处理
        elif isinstance(choices, str) and len(choices) == 1 and choices.isalpha():
            return [choices]
        return None
    except:
        # 直接解析各种格式
        # 尝试处理 \boxed{A,B} 这种格式
        if ',' in content:
            choices = [choice.strip() for choice in content.split(',')]
            if all(len(choice) == 1 and choice.isalpha() for choice in choices):
                return choices
        # 尝试处理 \boxed{A} 这种单选格式
        elif content.strip("'\"").isalpha() and len(content.strip("'\"")) == 1:
            return [content.strip("'\"")]
        return None


def is_choice_question(text: str) -> bool:
    """判断是否为选择题类型"""
    if not extract_boxed_content(text):
        return False
        
    content = extract_boxed_content(text)
    try:
        # 尝试评估内容
        parsed = eval(content)
        # 如果是列表且元素都是单个字母字符串，则可能是选择题
        if isinstance(parsed, list):
            if all(isinstance(x, str) and len(x) == 1 and x.isalpha() for x in parsed):
                return True
        # 如果是单个字符串且是单个字母，也可能是选择题
        elif isinstance(parsed, str) and len(parsed) == 1 and parsed.isalpha():
            return True
        return False
    except:
        # 直接检查内容是否为选择题格式
        # 检查 A,B 格式
        if ',' in content:
            choices = [choice.strip() for choice in content.split(',')]
            if all(len(choice) == 1 and choice.isalpha() for choice in choices):
                return True
        # 检查单个字母格式
        elif re.match(r"^\s*['\"]\s*[A-Z]\s*['\"]$", content) or re.match(r"^\s*[A-Z]\s*$", content):
            return True
        return False


def is_basic_question(text: str) -> bool:
    """判断是否为基本问题类型，基本问题的\boxed内容是整数或浮点数"""
    content = extract_boxed_content(text)
    if not content:
        return False
        
    try:
        # 尝试评估内容是否为数字（整数或浮点数）
        parsed = eval(content)
        # 如果是整数或浮点数，则是基本问题
        if isinstance(parsed, (int, float)):
            return True
        return False
    except:
        # 直接检查内容是否为数字格式
        # 移除可能的空格
        content = content.strip()
        # 检查是否为整数格式
        if content.isdigit():
            return True
        # 检查是否为浮点数格式
        try:
            float(content)
            return True
        except ValueError:
            return False


def choice_answer_reward(predict_str: str, ground_truth_str: str) -> float:
    """计算选择题答案的准确性"""
    pred_choices = extract_choice_answers(predict_str)
    gt_choices = extract_choice_answers(ground_truth_str)
    
    if pred_choices is None or gt_choices is None:
        return 0.0
    
    # 转换为集合以便于比较
    pred_set = set(choice.upper() for choice in pred_choices)
    gt_set = set(choice.upper() for choice in gt_choices)
    
    # 如果预测和真实都是空集，返回满分
    if not gt_set and not pred_set:
        return 1.0
    
    # 如果其中一个是空集而另一个不是，返回0分
    if not gt_set or not pred_set:
        return 0.0
    
    # 计算准确率和召回率
    true_positives = len(pred_set.intersection(gt_set))
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gt_set) if gt_set else 0.0
    
    # 使用F1-score作为最终得分
    if precision + recall == 0:
        return 0.0
        
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def extract_basic_answer(text: str) -> Union[int, float, None]:
    """从\boxed{}标签中提取基本问题答案（整数或浮点数）"""
    content = extract_boxed_content(text)
    if not content:
        return None
    try:
        # 尝试评估为数字
        parsed = eval(content)
        if isinstance(parsed, (int, float)):
            return parsed
        return None
    except:
        # 直接解析数字格式
        content = content.strip()
        try:
            # 尝试转换为浮点数
            return float(content)
        except ValueError:
            return None


def basic_answer_reward(predict_str: str, ground_truth_str: str) -> float:
    """计算基本问题答案的准确性"""
    pred_answer = extract_basic_answer(predict_str)
    gt_answer = extract_basic_answer(ground_truth_str)
    
    if pred_answer is None or gt_answer is None:
        return 0.0
    
    # 对于整数和浮点数都使用相对误差计算准确性
    relative_error = abs(pred_answer - gt_answer) / max(1, abs(gt_answer))  # 避免除零
    
    # 根据相对误差计算得分
    # 当相对误差为0时得分为1，当相对误差≥1时得分为0
    return max(0, 1 - relative_error)


def ts_compute_score(predict_str: str, ground_truth_str: str) -> float:
    """
    计算总体评分
    predict_str: 模型预测的完整文本输出
    ground_truth_str: 标准答案的完整文本输出
    """
    # 检查是否为基本问题类型
    if is_basic_question(ground_truth_str):
        format_score = ts_format_reward(predict_str)
        basic_score = basic_answer_reward(predict_str, ground_truth_str)
        
        # 基本问题的权重分配
        weights = {
            'format': 0.2,
            'basic': 0.8
        }
        
        final_score = (
            weights['format'] * format_score +
            weights['basic'] * basic_score
        )
        accuracy = basic_score
        format = format_score
    # 检查是否为选择题类型
    elif is_choice_question(ground_truth_str):
        format_score = ts_format_reward(predict_str)
        choice_score = choice_answer_reward(predict_str, ground_truth_str)
        
        # 选择题的权重分配
        weights = {
            'format': 0.2,
            'choice': 0.8
        }
        
        final_score = (
            weights['format'] * format_score +
            weights['choice'] * choice_score
        )
        accuracy = choice_score
        format = format_score
    else:
        # 原始时间序列分析问题的评分
        format_score = ts_format_reward(predict_str)
        period_score = period_reward(predict_str, ground_truth_str)
        recon_score = recon_reward(predict_str, ground_truth_str)
        anomaly_score = anomaly_intervals_reward(predict_str, ground_truth_str)
        
        # 权重分配 wo COT
        weights = {
            'format': 0.2,
            'period': 0,
            'recon': 0,
            'anomaly': 0.8
        }
        
        final_score = (
            weights['format'] * format_score +
            weights['period'] * period_score +
            weights['recon'] * recon_score +
            weights['anomaly'] * anomaly_score
        )
        accuracy = anomaly_score
        format = format_score
    
    return {
        "overall": final_score,
        "format": format,
        "accuracy": accuracy,
    }
