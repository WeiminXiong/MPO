import os
import json
import argparse
import re
from tqdm import tqdm

def extract_workflow(text):
    """Extract workflow from text"""
    try:
        workflow = re.findall(r'<workflow>(.*?)</workflow>', text, re.DOTALL)[0].strip()
    except:
        workflow = text.strip()
    return workflow

def split_metaplans(env_type, input_path, output_dir, flow_cnt):
    """Split metaplan samples into multiple files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    print(f"Loaded {len(data)} tasks")
    
    # Create output files for each flow
    for flow_idx in range(flow_cnt):
        output_data = []
        
        for item in tqdm(data, desc=f"Processing flow {flow_idx}"):
            task_id = item.get("id", None)
            task = item.get("task", "")
            
            # Process workflow, ensure we get the correct workflow
            if isinstance(item.get("workflow", ""), list) and flow_idx < len(item["workflow"]):
                workflow = extract_workflow(item["workflow"][flow_idx])
            else:
                workflow = extract_workflow(item.get("workflow", ""))
            
            output_item = {
                "task": task,
                "workflow": workflow
            }
            
            if task_id is not None:
                output_item["id"] = task_id
                
            output_data.append(output_item)
        
        # Write output file
        output_path = os.path.join(output_dir, f"{env_type}_metaplan_train_sampled_{flow_idx}.jsonl")
        with open(output_path, 'w') as f:
            for item in output_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(output_data)} tasks to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split metaplan samples into multiple files")
    parser.add_argument("--env", type=str, choices=["alfworld", "sciworld"], required=True,
                        help="Environment type: alfworld or sciworld")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Input file path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--flow_cnt", type=int, required=True,
                        help="Number of flows")
    
    args = parser.parse_args()
    
    split_metaplans(args.env, args.input_path, args.output_dir, args.flow_cnt)
