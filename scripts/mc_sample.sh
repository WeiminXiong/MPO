#!/bin/bash

flow_num=5
mc_num=5
node_num=4
node_start=0
start_port=8000
node_num=8 # The number of GPUs

# Task configuration
TASK_TYPE="sciworld"
INCORPORATION_TYPE="query"  # query, observation, thought
METAPLAN_GENERATOR="***************" # The sft-initialized model for metaplan generation
GENERATOR_SAVE_PATH="***************" # The save path of the sft-initialized model
EXPLORER_MODEL="***************" # The explorer model for metaplan quality evaluation
EXPLORER_SAVE_PATH="***************" # The save path of the explorer model
# Function to kill all background processes
cleanup() {
    echo "Caught Ctrl-C, killing all background processes..."
    pkill -P $$
    exit 1
}
trap cleanup SIGINT

# Sample Metaplans
echo "Start sampling metaplans..."
CUDA_VISIBLE_DEVICES=0 vllm serve ${GENERATOR_SAVE_PATH}/${METAPLAN_GENERATOR} --port 8000 > logs/sample_metaplan_generator.log 2>&1 &
sleep 60
python scripts/generate_metaplan.py \
    --task_type ${TASK_TYPE} \
    --input_file data/${TASK_TYPE}/${TASK_TYPE}_train_tasks.jsonl \
    --output_file samples/${TASK_TYPE}_metaplan_train_sampled.jsonl \
    --sample_num ${flow_num} \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY \
    --model_name ${GENERATOR_SAVE_PATH}/${METAPLAN_GENERATOR} \
    --temperature 0.8

echo "Finish sampling metaplans."

# Kill the metaplan generator
pkill -P $$

# Split metaplans into multiple files
python scripts/split_metaplans_for_sample.py \
    --env ${TASK_TYPE} \
    --input_path samples/${TASK_TYPE}_metaplan_train_sampled.jsonl \
    --output_dir samples \
    --flow_cnt ${flow_num}

# Meta-plan quality evaluation
echo "Start meta-plan quality evaluation..."

# Start task completion agents
base_urls=()
for (( i=0; i<$node_num; i=i+1 )); do
    base_urls+=("http://localhost:$((start_port+i))/v1")
    CUDA_VISIBLE_DEVICES=$((start_node+i)) vllm serve $EXPLORER_SAVE_PATH/$EXPLORER_MODEL --port $((start_port+i)) > logs/${TASK_TYPE}_${EXPLORER_MODEL}_${i}.log 2>&1 &
done

# Start task completion agents
for (( i=0; i<$flow_num; i=i+1 )); do
    for (( j=0; j<$mc_num; j=j+1 )); do
        python main.py \
            --exp_config ${exp_config} \
            --agent_config explorer \
            --split train \
            --metaplan_type none \
            --incorporation_type ${INCORPORATION_TYPE} \
            --api_base ${base_urls[$(((i*mc_num+j)%node_num))]} \
            --api_key "EMPTY" \
            --output_dir samples/${TASK_TYPE}_metaplan_mc/metaplan-$i-run-$j &
    done
done

wait

pkill -P $$

# Construct metaplan pairs
python scripts/construct_metaplan_pairs.py \
    --env ${TASK_TYPE} \
    --metaplan_path samples \
    --sample_dir samples/${TASK_TYPE}_metaplan_mc \
    --output_path samples/${TASK_TYPE}_metaplan_preference_pairs.json \
    --flow_cnt ${flow_num} \
    --run_cnt ${mc_num}

echo "Finish constructing metaplan pairs."