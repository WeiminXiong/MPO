import os
import json
import argparse

import numpy as np
from tqdm import tqdm
from template import ALFWORLD_TEMPLATE, SCIWORLD_TEMPLATE


def construct_metaplan_pairs(env_type, metaplan_path, sample_dir, output_path, flow_cnt, run_cnt):
    if env_type == "alfworld":
        task_cnt = 3553
        template = ALFWORLD_TEMPLATE
        
        # Initialize task ID list
        ids = list(range(task_cnt))
        
        # Load metaplans
        metaplans = []
        for flow_idx in range(flow_cnt):
            with open(f"{metaplan_path}/{env_type}_metaplan_train_sampled_{flow_idx}.jsonl") as fr:
                raw = fr.readlines()
                cur_metaplans = [json.loads(w) for w in raw]
            metaplans.append(cur_metaplans)
        
        
    elif env_type == "sciworld":
        template = SCIWORLD_TEMPLATE
        
        # Load task IDs
        with open(f"{metaplan_path}/sciworld_metaplan_train_sampled.jsonl") as fr:
            raw = fr.readlines()
            task_data = [json.loads(w) for w in raw]
            ids = [w["id"] for w in task_data]
            task_cnt = len(ids)
        
        # Load metaplans
        metaplans = []
        for flow_idx in range(flow_cnt):
            with open(f"{metaplan_path}/sciworld_metaplan_train_sampled_{flow_idx}.jsonl") as fr:
                raw = fr.readlines()
                cur_metaplans = [json.loads(w) for w in raw]
            cur_metaplans = {w["id"]: w for w in cur_metaplans}
            metaplans.append(cur_metaplans)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

    # Calculate success rates
    all_success_rate = []
    for flow_idx in range(flow_cnt):
        success_rate = []
        for task_idx in tqdm(ids):
            cur_metric = []
            for run_idx in range(run_cnt):
                file_path = os.path.join(sample_dir, f"metaplan-{flow_idx}-run-{run_idx}", f"{task_idx}.json")
                raw = json.load(open(file_path))
                cur_metric.append(raw[-1]['reward'])
            success_rate.append(sum(cur_metric) / len(cur_metric))
        all_success_rate.append(success_rate)

    all_success_rate = np.array(all_success_rate)
    max_idx = np.argmax(all_success_rate, axis=0)
    min_idx = np.argmin(all_success_rate, axis=0)

    # Build preference pairs
    res = []
    for idx in range(task_cnt):
        if max_idx[idx] == min_idx[idx]:
            continue
            
        if env_type == "alfworld":
            task_id = idx
            chosen_metaplan = metaplans[max_idx[idx]][idx]
            rejected_metaplan = metaplans[min_idx[idx]][idx]
        else:  # sciworld
            task_id = ids[idx]
            chosen_metaplan = metaplans[max_idx[idx]][task_id]
            rejected_metaplan = metaplans[min_idx[idx]][task_id]
            
        pair = {
            "conversations": [
                {
                    "from": "human",
                    "value": template.format(task=chosen_metaplan["task"]),
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": f"<workflow>\n{chosen_metaplan['workflow']}\n</workflow>",
            },
            "rejected": {
                "from": "gpt",
                "value": f"<workflow>\n{rejected_metaplan['workflow']}\n</workflow>",
            }
        }
        
        # For SciWorld, add reward information
        if env_type == "sciworld":
            pair["reward"] = {
                "chosen": all_success_rate[max_idx[idx]][idx],
                "rejected": all_success_rate[min_idx[idx]][idx],
            }
            
        res.append(pair)

    # Save results
    json.dump(res, open(output_path, "w"), indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct metaplan preference pairs")
    parser.add_argument("--env", type=str, choices=["alfworld", "sciworld"], required=True,
                        help="Environment type: alfworld or sciworld")
    parser.add_argument("--metaplan_path", type=str, required=True,
                        help="Path to the sampled metaplans")
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Directory of sample agent task completions")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the constructed preference pairs")
    parser.add_argument("--flow_cnt", type=int, required=True,
                        help="Number of flows")
    parser.add_argument("--run_cnt", type=int, required=True,
                        help="Number of runs")
    
    args = parser.parse_args()
    
    construct_metaplan_pairs(args.env, args.metaplan_path, args.sample_dir, args.output_path, args.flow_cnt, args.run_cnt) 