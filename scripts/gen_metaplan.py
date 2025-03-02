import os
import re
import json
import asyncio
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from time import sleep
import argparse
from template import ALFWORLD_TEMPLATE, SCIWORLD_TEMPLATE

class MetaplanGenerator:
    def __init__(self, task_type="sciworld", base_url="http://localhost:8001/v1", api_key="EMPTY", model_name="llama3.1-8b", temperature=0):
        self.task_type = task_type
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.template = self._get_template()
        self.temperature = temperature
    def _get_template(self):
        if self.task_type == "sciworld":
            return SCIWORLD_TEMPLATE
        else:
            return ALFWORLD_TEMPLATE

    async def async_query_openai(self, query, item_id, sample_num=1):
        n = sample_num
        aclient = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        try_times = 3
        for i in range(try_times):
            try:
                completion = await aclient.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=512,
                    n=n,
                    temperature=self.temperature,
                )
                res = [completion.choices[i].message.content for i in range(n)]
                break
            except Exception as e:
                print(e)
                res = "ERROR"
                if i < try_times-1:
                    sleep(60)
        
        return {"res": res, "id": item_id}

    def parse_workflow(self, res):
        try:
            res_thought = re.findall(r'<workflow>(.*?)</workflow>', res, re.DOTALL)[0].strip()
        except:
            res_thought = res
        return res_thought

    async def generate_metaplans(self, input_file, output_file, sample_num=1, batch_size=10):
        # 读取输入数据
        raw = [json.loads(line) for line in open(input_file)]
        
        # 检查已完成的任务
        done_ids = set()
        if os.path.exists(output_file):
            for line in open(output_file):
                item = json.loads(line)
                done_ids.add(item['id'])

        for i in range(0, len(raw), batch_size):
            batch = raw[i:i+batch_size]
            new_batch = []
            queries = []
            
            for idx, item in enumerate(batch):
                item_id = item["id"]
                if item_id in done_ids:
                    continue
                    
                task = item["task"]
                new_batch.append((item_id, task))
                queries.append((self.template.format(task=task), item_id))

            if not queries:
                continue

            # 执行批量查询
            tasks = [self.async_query_openai(q, idx) for q, idx in queries]
            results = []
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                results.append(await task)

            # 处理结果并写入文件
            for item_id, task in new_batch:
                for res in results:
                    if res['id'] == item_id:
                        workflows = [self.parse_workflow(r) for r in res['res']]
                        cur = {
                            "id": item_id,
                            "task": task,
                            "workflow": workflows
                        }
                        with open(output_file, "a") as fw:
                            fw.write(json.dumps(cur) + "\n")
                        break

async def main(task_type, input_file, output_file, sample_num, base_url, api_key, model_name):
    # sample metaplans
    generator = MetaplanGenerator(task_type, base_url, api_key, model_name)
    await generator.generate_metaplans(
        input_file,
        output_file,
        sample_num
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, required=True,
                        help="Task type: sciworld or alfworld")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input file path")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path")
    parser.add_argument("--sample_num", type=int, required=True, default=1,
                        help="Sample size")
    parser.add_argument("--base_url", type=str, required=True, default="http://localhost:8001/v1",
                        help="Base URL")
    parser.add_argument("--api_key", type=str, required=True, default="EMPTY",
                        help="API key")
    parser.add_argument("--model_name", type=str, required=True, default="llama3.1-8b",
                        help="metaplan generation model name")
    parser.add_argument("--temperature", type=float, required=True, default=0,
                        help="temperature")
    args = parser.parse_args()
    asyncio.run(main(args.task_type, args.input_file, args.output_file, args.sample_num, args.base_url, args.api_key, args.model_name, args.temperature)) 