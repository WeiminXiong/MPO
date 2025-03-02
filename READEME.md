# Boosting LLM Agents with Meta Plan Optimization
</div> 
<div align="center">
<a href=""><img src="assets/paper-page-xl.svg" alt="Paper page"></a>
<a href="https://huggingface.co/datasets/xwm/Meta_Plan_Optimization"><img src="assets/dataset-on-hf-xl.svg" alt="Dataset on HF"></a>
<a href="https://huggingface.co/xwm/sciworld_workflow_mpo"><img src="assets/model-on-hf-xl.svg" alt="Model on HF"></a>
</div>

This repository contains the code for the paper "Boosting LLM Agents with Meta Plan Optimization"

<p align="center">
<img src=assets/main.png width=700/>
</p>

In this work, we introduce the **Meta Plan Optimization (MPO)** framework, designed to enhance agent planning capabilities by directly integrating explicit guidance. Unlike previous methods that depend on complex knowledge—often requiring extensive human effort or lacking quality assurance—MPO leverages high-level general guidance through meta plans. This approach not only assists agents in planning but also enables continuous optimization of meta plans based on feedback from the agent's task execution.  


## 🧩 Structure of This Project
There are eight main folders in this project: `agents`, `configs`, `data`, `envs`, `prompt`, `scripts`, `tasks`, `utils`.

`agents`: code for the agents

`configs`: configuration files for the experiments

`data`: data for the experiments

`envs`: code for the environments

`prompt`: prompt templates

`scripts`: script for dataset construction and meta plan generation

`tasks`: code for the tasks

`utils`: utility functions


## 🛠️ Setup

```bash
git clone https://github.com/WeiminXiong/MPO.git
cd MPO
conda create -n mpo python=3.10
conda activate mpo
pip install -r requirements.txt
bash download_data.sh
```

## 🚀 Quick Start
To evaluate the effectiveness of MPO-optimized meta plans on baseline models, directly run the following bash script:
```bash
bash run_experiment.sh
```
The script performs the following steps:

1. configure the experiment parameters in `run_experiment.sh`
2. launch the model server
3. run the experiment

## 🎮 Dataset Construction
To generate training data for the DPO optimization phase of the meta planner, run the following bash script.
```bash
bash scripts/mc_sample.sh
```
The script performs the following steps:
1. configure the experiment parameters in `scripts/mc_sample.sh`
2. sample metaplans from the SFT-initialized metaplan generator
3. let the explorer agent to evaluate the quality of the sampled metaplans
4. generate training data for the DPO optimization phase of the meta planner

For more details about the dataset construction, please refer to the `scripts` directory.

## 📖 Citation

If you find this repo helpful, please cite out paper:

```
```