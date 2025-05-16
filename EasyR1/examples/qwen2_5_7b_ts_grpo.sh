set -x


MODEL_PATH=savepath_stage2  # Replace it with savepath_stage2
FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=examples/config_ts_vlm.yaml \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=4 \
    trainer.experiment_name=rl_after_sft_vision_tower \
    worker.rollout.gpu_memory_utilization=0.5 \
    data.train_files=./data/ts_train_image_normal.parquet \
    data.val_files=./data/ts_test_image_normal.parquet \
    trainer.save_checkpoint_path=savepath_stage3
