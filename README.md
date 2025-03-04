<h1 align="center">
<img src="assets/planner.png" width="100" alt="rho-logo" />
<br>
Boosting LLM Agents with Meta Plan Optimization
</h1>

<div align="center">

![](https://img.shields.io/badge/Paper-arXiv-red)
![](https://img.shields.io/badge/Model-Released-blue)
![](https://img.shields.io/badge/Code%20License-Apache%202.0-green)

</div>

<p align="center">
  <a href=""><b>[📜 Arxiv]</b></a> •
  <a href="https://huggingface.co/xwm/ALFWorld-MPO"><b>[🤗 Models]</b></a> •
  <a href="https://github.com/WeiminXiong/MPO"><b>[🐱 GitHub]</b></a>
</p>

This repository contains the code for the paper "Boosting LLM Agents with Meta Plan Optimization"

<p align="center">
<img src=assets/main.png width=700/>
</p>

In this work, we introduce the **Meta Plan Optimization (MPO)** framework, designed to enhance agent planning capabilities by directly integrating explicit guidance. Unlike previous methods that depend on complex knowledge—often requiring extensive human effort or lacking quality assurance—MPO leverages high-level general guidance through meta plans. This approach not only assists agents in planning but also enables continuous optimization of meta plans based on feedback from the agent's task execution.  


## 🔥 News

- [2025/03/06] 🔥🔥🔥 MPO-optimized meta planner released at 🤗 HuggingFace! 
    - Llama-3.1-70B-Instruct, enhanced with the MPO-optimized meta planner ([ALFWorld-MPO](https://huggingface.co/xwm/ALFWorld-MPO) and [SciWorld-MPO](https://huggingface.co/xwm/SciWorld-MPO)), achieved an average accuracy of 83.1 on ALFWorld and SciWorld, setting a new state-of-the-art (SOTA) performance.
    - Llama-3.1-8B-Instruct + MPO achieved an average performance of 53.6, outperforming GPT-4o-mini by a significant margin with a 30.1% improvement.
- [2025/03/04] MPO paper and repo released.


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

## 📖 Citation

If you find this repo helpful, please cite out paper:

```
```