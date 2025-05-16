import numpy as np
import pandas as pd
import uuid
from typing import List, Dict, Any, Tuple, Optional
import random
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import io
import base64
import os
import json
from scipy.signal import spectrogram, stft

random.seed(42)
np.random.seed(42)
ONE_SHOT = False
FEW_SHOT = False
TYPE = "image"  # image or text or text_and_image
COT = False
PLOT_TYPE = "PLOT1"


class TrendType(Enum):
    LINEAR = "linear"


class AnomalyType(Enum):
    SPIKE = "spike"
    TREND = "trend"
    FREQUENCY = "frequency"
    LEVEL_SHIFT = "level_shift"


@dataclass
class TimeSeriesFeatures:
    """Features of a time series"""

    length: int
    has_change_point: bool
    change_point: Optional[int]
    trend_type: Optional[TrendType]
    trend_type2: Optional[TrendType]
    periods: List[int]
    has_anomalies: bool
    anomalies: List[Dict[str, Any]]
    mean: float
    std: float
    max_value: float
    min_value: float
    base_trend: np.ndarray
    base_seasonal: np.ndarray
    # Additional features - Trend features
    trend_direction: str  # 'increase', 'decrease', 'stable'
    trend_slope: float  # Trend slope
    trend_intercept: float  # Trend intercept
    trend_strength: float  # Trend strength (0-1) - Contribution of trend to total time series
    # Additional features - Seasonality features
    seasonality_strength: float  # Seasonality strength (0-1) - Contribution of seasonality to total time series
    main_period: int  # Main period
    seasonal_amplitudes: List[float]  # Amplitudes of seasonal components
    seasonal_phases: List[float]  # Phases of seasonal components
    freq_multiples: List[float]  # Frequency multiples
    # Additional features - Noise features
    noise_level: float  # Noise level
    noise_to_signal_ratio: float  # Noise to signal ratio
    # Additional features - Statistical features
    skewness: float  # Skewness
    kurtosis: float  # Kurtosis
    acf_values: List[float]  # Autocorrelation coefficients
    stationarity: float  # Stationarity measure (0-1)
    # Additional features - Anomaly related
    level_shifts: List[Dict[str, Any]]  # Level shift details
    spikes: List[Dict[str, Any]]  # Spike details
    trend_changes: List[Dict[str, Any]]  # Trend change details
    freq_changes: List[Dict[str, Any]]  # Frequency change details


def plot_time_series(
    ts: np.ndarray, features: TimeSeriesFeatures
) -> List[Dict[str, Any]]:
    """Plot time series and return a list of image dictionaries"""
    if PLOT_TYPE == "PLOT1":
        plt.figure(figsize=(12, 6))
        plt.plot(ts, label="Time Series", color="blue")
        plt.title("Time Series")
        plt.xlabel("Index")
        plt.ylabel("Value")
    else:
        raise Exception("Not Implemented")

    # Generate unique filename
    img_filename = f"ts_{uuid.uuid4().hex[:8]}.png"
    img_path = f"./SFT_IMAGE_{PLOT_TYPE}/{img_filename}"
    # Save to local filesystem
    plt.savefig(img_path, dpi=100, bbox_inches="tight")
    plt.close()

    return [img_path]


def round_ts(ts: np.ndarray) -> np.ndarray:
    """Round time series to 3 decimal places"""
    return np.round(ts, 3)


def generate_trend_segment(length: int, trend_type: TrendType) -> np.ndarray:
    """Generate a segment of trend with specified type"""
    if trend_type == TrendType.LINEAR:
        trend_direction = random.choices(
            ["increase", "decrease", "stable"], weights=[0.2, 0.2, 0.6], k=1
        )[0]
        if trend_direction == "increase":
            slope = random.uniform(0.5, 4)
        elif trend_direction == "decrease":
            slope = random.uniform(-4, -0.5)
        else:
            slope = 0
        intercept = random.uniform(-2, 2)
        trend = slope * np.arange(length) + intercept
    return trend


def generate_time_series(length: int = 100) -> Tuple[np.ndarray, TimeSeriesFeatures]:
    """Generate time series with specified features"""
    has_change_point = False
    change_point = None
    trend_type = random.choice(list(TrendType))
    trend_type2 = None

    # Generate trend direction and slope
    trend_direction = random.choices(
        ["increase", "decrease", "stable"], weights=[0.2, 0.2, 0.6], k=1
    )[0]
    if trend_direction == "increase":
        slope = random.uniform(0.5, 4)
    elif trend_direction == "decrease":
        slope = random.uniform(-4, -0.5)
    else:
        slope = 0
    intercept = random.uniform(-2, 2)

    trend = slope * np.arange(length) + intercept

    trend_std = np.abs(np.mean(trend))
    max_period = random.randint(10, min(length // 2, 50))
    base_freq = 1 / max_period

    num_freqs = random.randint(1, 5)
    possible_freq_multiples = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    freq_multiples = [1]
    additional_freqs = random.sample(possible_freq_multiples, num_freqs - 1)
    freq_multiples.extend(additional_freqs)
    freq_multiples.sort()
    periods = [round(max_period / multiple) for multiple in freq_multiples]

    # Save amplitudes and phases of seasonal components
    seasonal_amplitudes = []
    seasonal_phases = []

    seasonality = np.zeros(length)
    for freq_multiple in freq_multiples:
        phase = random.uniform(0, 2 * np.pi)
        amplitude = random.uniform(0.2, 0.4) * trend_std / (freq_multiple**0.5)
        seasonality += amplitude * np.sin(
            2 * np.pi * freq_multiple * base_freq * np.arange(length) + phase
        )

        # Save amplitude and phase for each frequency
        seasonal_amplitudes.append(round(amplitude, 4))
        seasonal_phases.append(round(phase, 4))

    noise_amplitude = random.uniform(0.05, 0.1) * trend_std
    noise = np.random.normal(0, noise_amplitude, length)

    ts = trend + seasonality + noise

    # Calculate strength of trend, seasonality and noise
    total_variance = np.var(ts) if np.var(ts) > 0 else 1
    trend_strength = np.var(trend) / total_variance if total_variance > 0 else 0
    seasonality_strength = (
        np.var(seasonality) / total_variance if total_variance > 0 else 0
    )
    noise_level = np.var(noise) / total_variance if total_variance > 0 else 0
    noise_to_signal_ratio = (
        np.var(noise) / (np.var(trend + seasonality))
        if np.var(trend + seasonality) > 0
        else 1
    )

    # Calculate statistical features
    skewness = (
        float(np.mean(((ts - np.mean(ts)) / np.std(ts)) ** 3)) if np.std(ts) > 0 else 0
    )
    kurtosis = (
        float(np.mean(((ts - np.mean(ts)) / np.std(ts)) ** 4)) if np.std(ts) > 0 else 0
    )

    # Calculate autocorrelation coefficients (up to 50 lags or half the sequence length, whichever is smaller)
    max_lag = min(50, length // 2)
    acf_values = []
    for lag in range(1, max_lag + 1):
        # Simple calculation of autocorrelation coefficient
        if np.std(ts) > 0:
            acf = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
            acf_values.append(round(float(acf), 4))
        else:
            acf_values.append(0)

    # Simple stationarity measure - based on variance ratio of first difference
    if np.var(ts) > 0:
        diff_ts = np.diff(ts)
        stationarity = 1 - (np.var(diff_ts) / np.var(ts))
    else:
        stationarity = 1.0

    # Store different types of anomalies separately
    anomalies = []
    level_shifts = []
    spikes = []
    trend_changes = []
    freq_changes = []

    if random.random() < 0.7:
        anomaly_counts = [1, 2, 3]
        anomaly_weights = [0.7, 0.25, 0.05]
        num_anomalies = random.choices(anomaly_counts, weights=anomaly_weights, k=1)[0]

        for _ in range(num_anomalies):
            anomaly_type = random.choice(list(AnomalyType))
            start_pos = random.randint(20, length - 20)

            if anomaly_type == AnomalyType.TREND:
                has_change_point = True
                change_point = start_pos

                current_value = ts[change_point]
                if change_point > 1:
                    current_slope = ts[change_point] - ts[change_point - 1]
                else:
                    current_slope = 0

                if random.random() < 1.1:
                    trend_type2 = trend_type
                    if random.random() < 0.5:
                        new_slope = (
                            current_slope
                            * random.uniform(1.5, 3)
                            * random.choice([1, -1])
                        )
                    else:
                        new_slope = (
                            current_slope
                            * random.uniform(1 / 3, 2 / 3)
                            * random.choice([1, -1])
                        )
                    new_trend = (
                        new_slope * np.arange(length - change_point) + current_value
                    )

                transition_length = min(10, length - change_point)
                weights = np.linspace(1, 0, transition_length)

                original_trend = ts - seasonality - noise

                for i in range(transition_length):
                    blend_weight = weights[i]
                    blended_trend = (
                        blend_weight * original_trend[change_point + i]
                        + (1 - blend_weight) * new_trend[i]
                    )
                    ts[change_point + i] = (
                        blended_trend
                        + seasonality[change_point + i]
                        + noise[change_point + i]
                    )

                if change_point + transition_length < length:
                    ts[change_point + transition_length :] = (
                        new_trend[transition_length:]
                        + seasonality[change_point + transition_length :]
                        + noise[change_point + transition_length :]
                    )

                anomaly_info = {
                    "type": AnomalyType.TREND,
                    "position": change_point - 3,
                    "length": 6,
                    "new_trend_type": trend_type2.value,
                    "same_type": trend_type2 == trend_type,
                }
                anomalies.append(anomaly_info)
                trend_changes.append(anomaly_info)

            elif anomaly_type == AnomalyType.SPIKE:
                spike_length = random.randint(1, 5)
                spike_magnitude = random.uniform(3, 5) * np.std(ts)
                if random.random() < 0.5:
                    spike_magnitude = -spike_magnitude

                ts[start_pos : start_pos + spike_length] += spike_magnitude
                anomaly_info = {
                    "type": AnomalyType.SPIKE,
                    "position": start_pos,
                    "length": spike_length,
                    "magnitude": round(spike_magnitude, 3),
                }
                anomalies.append(anomaly_info)
                spikes.append(anomaly_info)

            elif anomaly_type == AnomalyType.FREQUENCY:
                start_pos = random.randint(10, length - 60)
                anomaly_length = random.randint(max_period, min(55, length - start_pos))

                if random.random() < 1.1:
                    if random.random() < 0.5:
                        freq_multiplier = random.choice([3, 5])
                    else:
                        freq_multiplier = random.choice([0.2, 0.33])

                    new_seasonality = np.zeros(anomaly_length)
                    t = np.arange(start_pos, start_pos + anomaly_length)

                    original_seasonal = np.zeros(anomaly_length)
                    for freq_multiple, period in zip(freq_multiples, periods):
                        segment = seasonality[max(0, start_pos - period) : start_pos]
                        if len(segment) > 0:
                            amplitude = np.std(segment) * 2**0.5
                        else:
                            amplitude = 1.0

                        phase = 2 * np.pi * freq_multiple * base_freq * start_pos
                        new_seasonality += amplitude * np.sin(
                            2
                            * np.pi
                            * freq_multiple
                            * base_freq
                            * freq_multiplier
                            * (t - start_pos)
                            + phase
                        )

                    anomaly_subtype = "frequency"
                else:
                    amplitude_factor = (
                        random.uniform(2, 3)
                        if random.random() < 0.5
                        else random.uniform(0.3, 0.5)
                    )
                    original_segment = seasonality[
                        start_pos : start_pos + anomaly_length
                    ]
                    new_seasonality = original_segment * amplitude_factor
                    anomaly_subtype = "amplitude"

                transition_length = min(5, anomaly_length)
                weights_start = np.linspace(0, 1, transition_length)
                for i in range(transition_length):
                    blend_weight = weights_start[i]
                    ts[start_pos + i] = (
                        ts[start_pos + i]
                        - seasonality[start_pos + i]
                        + (
                            (1 - blend_weight) * seasonality[start_pos + i]
                            + blend_weight * new_seasonality[i]
                        )
                    )

                if anomaly_subtype == "frequency":
                    ts[
                        start_pos
                        + transition_length : start_pos
                        + anomaly_length
                        - transition_length
                    ] = (
                        ts[
                            start_pos
                            + transition_length : start_pos
                            + anomaly_length
                            - transition_length
                        ]
                        - seasonality[
                            start_pos
                            + transition_length : start_pos
                            + anomaly_length
                            - transition_length
                        ]
                        + new_seasonality[
                            transition_length : anomaly_length - transition_length
                        ]
                    )
                else:
                    ts[
                        start_pos
                        + transition_length : start_pos
                        + anomaly_length
                        - transition_length
                    ] = (
                        ts[
                            start_pos
                            + transition_length : start_pos
                            + anomaly_length
                            - transition_length
                        ]
                        - seasonality[
                            start_pos
                            + transition_length : start_pos
                            + anomaly_length
                            - transition_length
                        ]
                        + new_seasonality[
                            transition_length : anomaly_length - transition_length
                        ]
                    )

                weights_end = np.linspace(1, 0, transition_length)
                for i in range(transition_length):
                    blend_weight = weights_end[i]
                    pos = start_pos + anomaly_length - transition_length + i
                    if pos < len(ts):
                        ts[pos] = (
                            ts[pos]
                            - seasonality[pos]
                            + (
                                blend_weight
                                * new_seasonality[
                                    anomaly_length - transition_length + i
                                ]
                                + (1 - blend_weight) * seasonality[pos]
                            )
                        )

                anomaly_info = {
                    "type": AnomalyType.FREQUENCY,
                    "position": start_pos,
                    "length": anomaly_length,
                    "subtype": anomaly_subtype,
                    "freq_multiplier": (
                        freq_multiplier if anomaly_subtype == "frequency" else None
                    ),
                    "amplitude_factor": (
                        amplitude_factor if anomaly_subtype == "amplitude" else None
                    ),
                }
                anomalies.append(anomaly_info)
                freq_changes.append(anomaly_info)

            else:  # LEVEL_SHIFT
                shift_magnitude = random.uniform(2, 4) * np.std(ts)
                if random.random() < 0.5:
                    shift_magnitude = -shift_magnitude

                ts[start_pos:] += shift_magnitude

                anomaly_info = {
                    "type": AnomalyType.LEVEL_SHIFT,
                    "position": start_pos - 3,
                    "length": 6,
                    "magnitude": round(shift_magnitude, 3),
                }
                anomalies.append(anomaly_info)
                level_shifts.append(anomaly_info)

    # Normalize the time series
    min_ts = np.min(ts)
    max_ts = np.max(ts)
    ts = (ts - min_ts) / (max_ts - min_ts)

    features = TimeSeriesFeatures(
        length=length,
        has_change_point=has_change_point,
        change_point=change_point,
        trend_type=trend_type,
        trend_type2=trend_type2,
        periods=periods,
        has_anomalies=len(anomalies) > 0,
        anomalies=anomalies,
        mean=round(np.mean(ts), 3),
        std=round(np.std(ts), 3),
        max_value=round(np.max(ts), 3),
        min_value=round(np.min(ts), 3),
        base_trend=(trend - min_ts) / (max_ts - min_ts),
        base_seasonal=(seasonality - min_ts) / (max_ts - min_ts),
        trend_direction=trend_direction,
        trend_slope=round(slope, 4),
        trend_intercept=round(intercept, 4),
        trend_strength=round(trend_strength, 4),
        seasonality_strength=round(seasonality_strength, 4),
        main_period=max(periods),
        seasonal_amplitudes=seasonal_amplitudes,
        seasonal_phases=seasonal_phases,
        freq_multiples=freq_multiples,
        noise_level=round(noise_level, 4),
        noise_to_signal_ratio=round(noise_to_signal_ratio, 4),
        skewness=round(skewness, 4),
        kurtosis=round(kurtosis, 4),
        acf_values=acf_values,
        stationarity=round(stationarity, 4),
        level_shifts=level_shifts,
        spikes=spikes,
        trend_changes=trend_changes,
        freq_changes=freq_changes,
    )

    return round_ts(ts), features


def generate_question(ts: np.ndarray, features: TimeSeriesFeatures) -> Dict[str, Any]:
    """Generate a question based on time series features"""
    # 生成图像列表
    if not ONE_SHOT and not FEW_SHOT:
        background_info = (
            "Background Information:\n"
            "In time series analysis, we focus on four main types of anomalies:\n"
            "1. Spike Anomaly: Sudden significant increase or decrease in values\n"
            "2. Trend Anomaly: Sudden change in trend direction or slope\n"
            "3. Frequency Anomaly: Changes in periodicity or amplitude\n"
            "4. Level Shift: Sudden overall increase or decrease in baseline level\n\n"
        )
    elif ONE_SHOT:
        background_info = (
            "Background Information:\n"
            "In time series analysis, we focus on four main types of anomalies:\n"
            "1. Spike Anomaly: Sudden significant increase or decrease in values\n"
            "2. Trend Anomaly: Sudden change in trend direction or slope\n"
            "3. Frequency Anomaly: Changes in periodicity or amplitude\n"
            "4. Level Shift: Sudden overall increase or decrease in baseline level\n\n"
        )
    elif FEW_SHOT:
        background_info = (
            "Background Information:\n"
            "In time series analysis, we focus on four main types of anomalies:\n"
            "1. Spike Anomaly: Sudden significant increase or decrease in values\n"
            "2. Trend Anomaly: Sudden change in trend direction or slope\n"
            "3. Frequency Anomaly: Changes in periodicity or amplitude\n"
            "4. Level Shift: Sudden overall increase or decrease in baseline level\n\n"
        )
    if TYPE == "image":
        img_path = plot_time_series(ts, features)
        question = f"<image>Given time series visualization, analyze the time series and detect anomalies.\n\n"
    elif TYPE == "text":
        img_path = []
        question = (
            f"Input Data:\nTime Series = {ts.tolist()}\n\n"
            + f"Given the time series, analyze the time series and detect anomalies.\n\n"
        )
    elif TYPE == "text_and_image":
        img_path = plot_time_series(ts, features)
        question = (
            f"<image>Given time series visualization, analyze the time series and detect anomalies.\n\n"
            + f"Input Data:\nTime Series = {ts.tolist()}\n\n"
        )

    if COT:
        cot_prompt = "Let's think step by step.\n"
        problem = (
            f"{question}\n"
            f"{background_info}\n"
            "Requirements:\n"
            "1. Write analysis process within <think> </think> tags. Try to analyze directly, don't use python code.\n"
            "2. Write anomalous intervals (final answer) using python list format in \\boxed{}, for example: \\boxed{[[start1, end1], [start2, end2], ...]}\n"
            "3. If no anomalies detected, just output \\boxed{[]}\n"
            "4. Do not overlap anomalous intervals\n\n"
            f"{cot_prompt}\n"
        )
    else:
        cot_prompt = "Just output the final answer.\n"
        problem = (
            f"{question}\n"
            f"{background_info}\n"
            "Requirements:\n"
            "1. Write anomalous intervals (final answer) using python list format in \\boxed{}, for example: \\boxed{[[start1, end1], [start2, end2], ...]}\n"
            "2. If no anomalies detected, just output \\boxed{[]}\n"
            "3. Do not overlap anomalous intervals\n\n"
            f"{cot_prompt}\n"
        )


    trend_description = get_trend_description(features)
    seasonal_description = get_seasonal_description(features)
    statistical_description = get_statistical_description(features)
    anomaly_description = get_anomaly_description(features)

    if features.has_anomalies:
        anomaly_intervals = []
        for anomaly in features.anomalies:
            start = anomaly["position"]
            end = start + anomaly["length"]
            anomaly_intervals.append([start, end])

        reconstructed_ts = features.base_trend + features.base_seasonal
        reconstructed_ts = round_ts(reconstructed_ts)

        errors = np.abs(ts - reconstructed_ts)
        mse = np.mean(errors**2)
        threshold = np.std(ts) * 2
        if COT:
            solution = (
                "<think>\n"
                "1. Time Series Components Analysis:\n"
                f"{trend_description}\n"
                f"{seasonal_description}\n"
                f"{statistical_description}\n\n"
                "2. Anomaly Detection:\n"
                f"{anomaly_description}\n"
            )

            for i, (anomaly, interval) in enumerate(
                zip(features.anomalies, anomaly_intervals)
            ):
                solution += f"     * Anomaly {i+1}: {anomaly['type'].value} at interval [{interval[0]}, {interval[1]}]\n"

            solution += "</think>\n\n"
            solution += f"Final answer: \\boxed{{{anomaly_intervals}}}"
        else:
            solution = f"Final answer: \\boxed{{{anomaly_intervals}}}"
    else:
        reconstructed_ts = features.base_trend + features.base_seasonal
        reconstructed_ts = round_ts(reconstructed_ts)
        if COT:
            solution = (
                "<think>\n"
                "1. Time Series Components Analysis:\n"
                f"{trend_description}\n"
                f"{seasonal_description}\n"
                f"{statistical_description}\n\n"
                "2. Anomaly Detection:\n"
                "   - No significant deviations from normal pattern\n"
                "   - All variations within expected range\n"
                "</think>\n\n"
                "Final Answer: \\boxed{[]}"
            )
        else:
            solution = "Final Answer: \\boxed{[]}"

    return {
        "images": img_path,
        "problem": problem,
        "solution": solution,
        "anomalies": features.anomalies,
    }


def get_trend_description(features: TimeSeriesFeatures) -> str:
    trend_desc = f"   - Trend Type: {features.trend_type.value}\n"

    if features.trend_direction == "increase":
        trend_desc += f"   - Direction: Increasing (slope={features.trend_slope})\n"
    elif features.trend_direction == "decrease":
        trend_desc += f"   - Direction: Decreasing (slope={features.trend_slope})\n"
    else:
        trend_desc += f"   - Direction: Stable (slope≈0)\n"

    trend_desc += f"   - Trend Strength: {features.trend_strength:.2f} (contribution to total variance)\n"

    if features.has_change_point:
        trend_desc += (
            f"   - Change Point: Present at position {features.change_point}\n"
        )
    else:
        trend_desc += f"   - Change Point: None\n"

    return trend_desc


def get_seasonal_description(features: TimeSeriesFeatures) -> str:
    seasonal_desc = f"   - Main Period: \\period{{{features.main_period}}}\n"
    seasonal_desc += f"   - Seasonality Strength: {features.seasonality_strength:.2f}\n"

    if len(features.periods) > 1:
        freq_desc = ", ".join([str(p) for p in features.periods])
        seasonal_desc += f"   - Multiple Periods Present: {freq_desc}\n"

    return seasonal_desc


def get_statistical_description(features: TimeSeriesFeatures) -> str:
    stat_desc = f"   - Mean: {features.mean}, Std: {features.std}\n"
    stat_desc += f"   - Range: [{features.min_value}, {features.max_value}]\n"
    stat_desc += (
        f"   - Skewness: {features.skewness:.2f}, Kurtosis: {features.kurtosis:.2f}\n"
    )
    stat_desc += f"   - Noise Level: {features.noise_level:.2f}, Noise-to-Signal Ratio: {features.noise_to_signal_ratio:.2f}\n"
    stat_desc += f"   - Stationarity: {features.stationarity:.2f}\n"

    return stat_desc


def get_anomaly_description(features: TimeSeriesFeatures) -> str:
    if not features.has_anomalies:
        return "   - No anomalies detected\n"

    anomaly_desc = f"   - Analyze the time series in the figure.\n"

    if features.spikes:
        for i, spike in enumerate(features.spikes):
            anomaly_desc += (
                f"     * Spike anomaly at position {spike['position']} "
                f"with magnitude {spike['magnitude']:.2f}, length {spike['length']}\n"
            )

    if features.trend_changes:
        for i, change in enumerate(features.trend_changes):
            anomaly_desc += (
                f"     * Trend anomaly at position {change['position']} "
                f"with new trend type {change['new_trend_type']}\n"
            )

    if features.freq_changes:
        for i, change in enumerate(features.freq_changes):
            if change["subtype"] == "frequency":
                anomaly_desc += (
                    f"     * Frequency anomaly at position {change['position']} "
                    f"with frequency multiplier {change['freq_multiplier']}\n"
                )
            else:
                anomaly_desc += (
                    f"     * Amplitude anomaly at position {change['position']} "
                    f"with amplitude factor {change['amplitude_factor']:.2f}\n"
                )

    if features.level_shifts:
        for i, shift in enumerate(features.level_shifts):
            anomaly_desc += (
                f"     * Level shift at position {shift['position']} "
                f"with magnitude {shift['magnitude']:.2f}\n"
            )

    return anomaly_desc


def generate_shaplet_question(
    ts: np.ndarray, features: TimeSeriesFeatures
) -> Dict[str, Any]:
    """Generate multiple choice questions for time series anomaly detection"""
    # Basic background information
    background_info = (
        "Background Information:\n"
        "In time series analysis, we focus on four main types of anomalies:\n"
        "1. Spike Anomaly: Sudden significant increase or decrease in values\n"
        "2. Trend Anomaly: Sudden change in trend direction or slope\n"
        "3. Frequency Anomaly: Changes in periodicity or amplitude\n"
        "4. Level Shift: Sudden overall increase or decrease in baseline level\n\n"
    )

    # Set question description based on image type
    if TYPE == "image":
        img_path = plot_time_series(ts, features)
        question_intro = f"<image>Given the time series visualization, analyze the time series and detect anomalies.\n\n"
    elif TYPE == "text":
        img_path = []
        question_intro = f"Input Data:\nTime Series = {ts.tolist()}\n\nGiven the time series, analyze and detect anomalies.\n\n"
    elif TYPE == "text_and_image":
        img_path = plot_time_series(ts, features)
        question_intro = f"<image>Given the time series visualization, analyze the time series and detect anomalies.\n\nInput Data:\nTime Series = {ts.tolist()}\n\n"

    # Build problem description
    problem_description = (
        f"{question_intro}\n"
        f"{background_info}\n"
        "Question: What types of anomalies are present in this time series?\n"
    )

    # Set options
    options = {
        "A": "Spike Anomaly",
        "B": "Trend Anomaly",
        "C": "Frequency Anomaly",
        "D": "Level Shift",
        "E": "No anomalies",
    }

    # Add options to problem description
    for key, value in options.items():
        problem_description += f"{key}. {value}\n"

    if COT:
        cot_prompt = "Let's think step by step.\n"
        problem_description += (
            f"\nRequirements:\n"
            "1. Write analysis process within <think> </think> tags. Try to analyze directly, don't use python code.\n"
            "2. Write final answer in \\boxed{}, for example: \\boxed{A}\n"
            f"{cot_prompt}\n"
        )
    else:
        cot_prompt = "Just output the final answer.\n"
        problem_description += (
            f"\nRequirements:\n"
            "1. Write final answer in \\boxed{}, for example: \\boxed{A}\n"
            f"{cot_prompt}\n"
        )
    # Determine correct answers
    correct_answers = []
    if features.has_anomalies:
        anomaly_types = set()
        for anomaly in features.anomalies:
            if anomaly["type"] == AnomalyType.SPIKE:
                anomaly_types.add("A")
            elif anomaly["type"] == AnomalyType.TREND:
                anomaly_types.add("B")
            elif anomaly["type"] == AnomalyType.FREQUENCY:
                anomaly_types.add("C")
            elif anomaly["type"] == AnomalyType.LEVEL_SHIFT:
                anomaly_types.add("D")
        correct_answers = sorted(list(anomaly_types))
    else:
        correct_answers = ["E"]
    correct_answers = ", ".join(correct_answers)

    if COT:
        # need to add the analysis process
        solution = f"<think>\n"
        solution += f"Final Answer: \\boxed{{{correct_answers}}}\n"
        solution += "</think>\n\n"
    else:
        solution = f"Final Answer: \\boxed{{{correct_answers}}}\n"

    return {
        "images": img_path,
        "problem": problem_description,
        "solution": solution,
        "anomalies": features.anomalies,
    }


def generate_basic_question(
    ts: np.ndarray, features: TimeSeriesFeatures
) -> Dict[str, Any]:
    """Generate basic feature questions for time series data (max, min, length, index)"""

    if np.std(ts) > 0:
        ts_zscore = (ts - np.mean(ts)) / np.std(ts)
    else:
        ts_zscore = ts.copy()
    ts_zscore = round_ts(ts_zscore)  # 四舍五入保留3位小数

    # 设置问题类型
    question_types = [
        "max_value",  # 最大值问题
        "min_value",  # 最小值问题
        "length",  # 长度问题
        "index",  # 索引问题
    ]
    question_type = random.choice(question_types)

    # 根据问题类型设置问题描述
    if TYPE == "image":
        img_path = plot_time_series(ts_zscore, features)
        question_intro = f"<image>Given the time series visualization, "
    elif TYPE == "text":
        img_path = []
        question_intro = f"Input Data:\nTime Series = {ts_zscore.tolist()}\n\nGiven the time series, "
    elif TYPE == "text_and_image":
        img_path = plot_time_series(ts_zscore, features)
        question_intro = f"<image>Given the time series visualization, \n\nInput Data:\nTime Series = {ts_zscore.tolist()}\n\n"

    # 根据问题类型构建具体问题
    if question_type == "max_value":
        problem_description = (
            f"{question_intro}find the maximum value in this time series.\n\n"
        )
        correct_answer = round(float(np.max(ts_zscore)), 3)
    elif question_type == "min_value":
        problem_description = (
            f"{question_intro}find the minimum value in this time series.\n\n"
        )
        correct_answer = round(float(np.min(ts_zscore)), 3)
    elif question_type == "length":
        problem_description = (
            f"{question_intro}determine the length of this time series.\n\n"
        )
        correct_answer = features.length
    elif question_type == "index":
        index = random.randint(0, len(ts_zscore) - 1)
        value = round(float(ts_zscore[index]), 3)
        problem_description = f"{question_intro}find the index where the time series has value {value}.\n\n"

        # 检查该值是否在序列中有多个出现
        same_value_indices = np.where(np.abs(ts_zscore - value) < 1e-5)[0]
        if len(same_value_indices) > 1:
            # 如果有多个相同的值，随机选择一个作为答案
            index = int(random.choice(same_value_indices))

        correct_answer = int(index)

    if COT:
        cot_prompt = "Let's think step by step.\n"
        problem_description += (
            "Requirements:\n"
            "1. Write analysis process within <think> </think> tags. Try to analyze directly, don't use python code.\n"
            "2. Write final answer in \\boxed{}, for example: \\boxed{answer}\n"
            f"{cot_prompt}\n"
        )
    else:
        cot_prompt = "Just output the final answer.\n"
        problem_description += (
            "Requirements:\n"
            "1. Write final answer in \\boxed{}, for example: \\boxed{answer}\n"
            f"{cot_prompt}\n"
        )

    if COT:
        solution = f"<think>\n"
        if question_type == "max_value":
            solution += f"To find the maximum value in the time series, I need to identify the highest point in the data.\n"
            solution += f"The time series has been z-score standardized (mean 0, standard deviation 1).\n"
            solution += f"Looking at the data, the maximum value is {correct_answer}.\n"
        elif question_type == "min_value":
            solution += f"To find the minimum value in the time series, I need to identify the lowest point in the data.\n"
            solution += f"The time series has been z-score standardized (mean 0, standard deviation 1).\n"
            solution += f"Looking at the data, the minimum value is {correct_answer}.\n"
        elif question_type == "length":
            solution += f"To determine the length of the time series, I need to count the number of data points.\n"
            solution += f"The time series contains {correct_answer} data points.\n"
        elif question_type == "index":
            solution += f"To find the index where the value is {value}, I need to scan through the time series and find the position where this value occurs.\n"
            solution += f"This value occurs at index {correct_answer}.\n"

        solution += "</think>\n\n"
        solution += f"Final Answer: \\boxed{{{correct_answer}}}"
    else:
        solution = f"Final Answer: \\boxed{{{correct_answer}}}\n"

    return {
        "images": img_path,
        "problem": problem_description,
        "solution": solution,
        "question_type": question_type,  # 额外记录问题类型以便分析
        "anomalies": features.anomalies,
    }


def generate_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate time series Q&A data with visualizations"""
    data = []
    used_ids = set()

    for _ in range(num_samples):
        ts_length = random.randint(100, 200)
        ts, features = generate_time_series(ts_length)
        qa = generate_question(ts, features)

        while True:
            new_id = str(uuid.uuid4())
            if new_id not in used_ids:
                used_ids.add(new_id)
                break

        data.append(
            {
                "id": new_id,
                "images": qa["images"],
                "problem": qa["problem"],
                "solution": qa["solution"],
                "ts_length": ts_length, 
            }
        )
    return data


def generate_shaplet_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate multiple choice anomaly detection dataset for time series"""
    data = []
    used_ids = set()

    for _ in range(num_samples):
        ts_length = random.randint(100, 200)
        ts, features = generate_time_series(ts_length)
        qa = generate_shaplet_question(ts, features)

        while True:
            new_id = str(uuid.uuid4())
            if new_id not in used_ids:
                used_ids.add(new_id)
                break

        data.append(
            {
                "id": new_id,
                "images": qa["images"],
                "problem": qa["problem"],
                "solution": qa["solution"],
                "ts_length": ts_length,  # 添加问题类型标记
            }
        )
    return data


def generate_basic_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate basic feature questions dataset for time series (max, min, length, index)"""
    data = []
    used_ids = set()

    for _ in range(num_samples):
        ts_length = random.randint(100, 200)
        ts, features = generate_time_series(ts_length)
        qa = generate_basic_question(ts, features)

        while True:
            new_id = str(uuid.uuid4())
            if new_id not in used_ids:
                used_ids.add(new_id)
                break

        data.append(
            {
                "id": new_id,
                "images": qa["images"],
                "problem": qa["problem"],
                "solution": qa["solution"],
                "basic_question_type": qa["question_type"],  # 记录具体的基本问题类型
                "ts_length": ts_length,
            }
        )
    return data


def generate_mixed_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """生成混合类型的时间序列问题数据（随机选择普通问题或shapelet问题）"""
    data = []
    label = []
    used_ids = set()

    for _ in range(num_samples):
        ts_length = random.randint(100, 200)
        ts, features = generate_time_series(ts_length)

        # 随机决定是生成普通问题还是shapelet问题
        rand_val = random.random()
        if rand_val < 0.7:
            qa = generate_question(ts, features)
            question_type = "normal"
        elif rand_val < 0.85:
            qa = generate_shaplet_question(ts, features)
            question_type = "shaplet"
        else:
            qa = generate_basic_question(ts, features)
            question_type = "basic"

        while True:
            new_id = str(uuid.uuid4())
            if new_id not in used_ids:
                used_ids.add(new_id)
                break

        if qa["images"] is not None:
            data.append(
                {
                    "messages": [
                        {"role": "user", "content": qa["problem"]},
                        {"role": "assistant", "content": qa["solution"]},
                    ],
                    "images": qa["images"],
                }
            )
        else:
            data.append(
                {
                    "messages": [
                        {"role": "user", "content": qa["problem"]},
                        {"role": "assistant", "content": qa["solution"]},
                    ]
                }
            )
        label.append({"type": [item["type"].value for item in features.anomalies]})
    with open("./data/eval_label.json", "w") as f:
        json.dump(label, f)
    return data


def main():
    os.makedirs(f"./SFT_IMAGE_{PLOT_TYPE}", exist_ok=True)
    train_num = 5000
    test_num = 400
    eval_num = 2000

    # Data generation type, added mixed option
    data_type = os.environ.get(
        "DATA_TYPE", "mixed"
    )  # Default is mixed, can be normal, shaplet, or basic

    if data_type == "mixed":
        print("Generating mixed type time series questions (random normal and shapelet questions)...")

        print("Generating training data...")
        train_data = generate_mixed_data(train_num)
        print("Generating test data...")
        test_data = generate_mixed_data(test_num)
        print("Generating evaluation data...")
        eval_data = generate_mixed_data(eval_num)

        with open(f"./data/ts_train_{TYPE}_{data_type}_{PLOT_TYPE}.json", "w") as f:
            json.dump(train_data, f)
        with open(f"./data/ts_eval_{TYPE}_{data_type}_{PLOT_TYPE}.json", "w") as f:
            json.dump(eval_data, f)

    elif data_type == "shaplet":
        print("Generating shaplet format time series anomaly detection data...")
        print("Generating training data...")
        train_data = generate_shaplet_data(train_num)
        print("Generating evaluation data...")
        eval_data = generate_shaplet_data(eval_num)

        with open(f"./data/ts_train_{TYPE}_{data_type}.json", "w") as f:
            json.dump(train_data, f)
        with open(f"./data/ts_eval_{TYPE}_{data_type}.json", "w") as f:
            json.dump(eval_data, f)

    elif data_type == "basic":
        print("Generating basic question format time series data...")
        print("Generating training data...")
        train_data = generate_basic_data(train_num)
        print("Generating evaluation data...")
        eval_data = generate_basic_data(eval_num)

        with open(f"./data/ts_train_{TYPE}_{data_type}.json", "w") as f:
            json.dump(train_data, f)
        with open(f"./data/ts_eval_{TYPE}_{data_type}.json", "w") as f:
            json.dump(eval_data, f)
    else:
        print("Generating normal time series question data...")
        print("Generating training data...")
        train_data = generate_data(train_num)
        print("Generating evaluation data...")
        eval_data = generate_data(eval_num)

        with open(f"./data/ts_train_{TYPE}_{data_type}.json", "w") as f:
            json.dump(train_data, f)
        with open(f"./data/ts_eval_{TYPE}_{data_type}.json", "w") as f:
            json.dump(eval_data, f)


if __name__ == "__main__":
    main()
