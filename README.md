# ViTs: Empowering Time Series Anomaly Detection with Vision Language Models

Three stage chain-style fine-tuning strategy.

## Stage 1 && Stege 2

### Download Model

Download Qwen/Qwen2.5-VL-7B-Instruct from Huggingface or ModelScope

### Environment
```
cd ChatAD
conda create -n llama-factory python=3.10.16
conda activate llama-factory
pip install -e ".[torch,metrics]"
pip install deepspeed==0.16.4
pip install flash-atten.whl # Download from website.
```

### Data Generator

1. Stage 1 Data (Time Series Description QAs)

```
python generate_vit_data.py
```
Data saved as ./data/ts_train_image_description_PLOT1.json

2. Stage 1 Data (TSAD QAs)
```
python generate_sft_data.py
```
Data saved as ./data/ts_train_image_mixed_PLOT1.json

### Training VLMs

```
/bin/bash train_stage1_stage2.sh
```


## Stage 3


### Environment
```
cd EasyR1
conda create -n easyr1 python=3.11.11
conda activate easyr1
pip install -e .
pip install vllm==0.8.3
pip install flash-atten.whl # Download from website.
```

### Data Generator

1. Stage 3 Data (TSAD QAs)

```
cd EasyR1/data
python generate_rl_dataset.py
python split_dataset.py --input ./ts_train_image_mixed.parquet
```


### Training VLMs

```
cd EasyR1
/bin/bash examples/qwen2_5_7b_ts_grpo.sh 
```
