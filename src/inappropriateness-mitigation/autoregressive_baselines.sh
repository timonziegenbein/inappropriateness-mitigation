#export CUDA_VISIBLE_DEVICES=3

LM_MODEL_LIST=(
    "EleutherAI/gpt-j-6B"
    "bigscience/bloom-7b1"
    "huggyllama/llama-7b"
    "facebook/opt-6.7b"
)

for model in "${LM_MODEL_LIST[@]}"
do
    python ./autoregressive_baselines.py --model_name $model --model_type lm --num_shots 0
done

for model in "${LM_MODEL_LIST[@]}"
do
    python ./autoregressive_baselines.py --model_name $model --model_type lm --num_shots 1
done

for model in "${LM_MODEL_LIST[@]}"
do
    python ./autoregressive_baselines.py --model_name $model --model_type lm --num_shots 4
done

for model in "${LM_MODEL_LIST[@]}"
do
    python ./autoregressive_baselines.py --model_name $model --model_type lm --num_shots 9
done

InstructLM_MODEL_LIST=(
    "../../data/models/instruction-finetuning/llama-7b-instruct"
    "../../data/models/instruction-finetuning/gpt-j-6b-instruct"
    "../../data/models/instruction-finetuning/bloom-7b1-instruct"
    "../../data/models/instruction-finetuning/opt-6.7b-instruct"
)

for model in "${InstructLM_MODEL_LIST[@]}"
do
    python ./autoregressive_baselines.py --model_name $model --model_type instruct --num_shots 0
done

for model in "${InstructLM_MODEL_LIST[@]}"
do
    python ./autoregressive_baselines.py --model_name $model --model_type instruct --num_shots 1
done

for model in "${InstructLM_MODEL_LIST[@]}"
do
    python ./autoregressive_baselines.py --model_name $model --model_type instruct --num_shots 4
done

for model in "${InstructLM_MODEL_LIST[@]}"
do
    python ./autoregressive_baselines.py --model_name $model --model_type instruct --num_shots 9
done
