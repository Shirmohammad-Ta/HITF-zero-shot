# Hybrid Instruction Tuning with Marginalization for Zero-Shot Reasoning in Language Models

This repository contains the code and resources for the paper:
 
 ## 🔬 Paper Reference
 **Title:"Hybrid Instruction Tuning with Marginalization for Zero-Shot Reasoning in Language Models"**
 **Author:** Shirmohammad Tavangari
 **My paper has been accepted at an IEEE conference!**
    
---

##  Overview

This project introduces **Hybrid Instruction Tuning (HITF)**, a method for improving zero-shot reasoning in large language models (LLMs) using:
- Dynamic example selection (via a learned selector)
- Marginalization over intermediate responses
- Few-shot prompting without retraining

It supports both **fine-tuning** and **prompt-only inference** with OpenAI models (GPT-3.5 / GPT-4) and HuggingFace models (e.g. T5, LLaMA).

---

##  Repository Structure

```bash
HITF-zero-shot/
├── src/                   # Source code: models, training, evaluation
├── data/                  # Example datasets (SuperGLUE, arithmetic)
├── prompts/               # Few-shot prompt examples
├── results/               # Output predictions and evaluation logs
├── paper/                 # PDF of the submitted paper (optional)
├── requirements.txt       # Python dependencies
└── README.md              # You are here!
```

---

##  Installation

```bash
pip install -r requirements.txt
```

Dependencies include:
- transformers
- torch
- openai
- scikit-learn
- peft
- tqdm

---

##  Usage

###  1. Evaluate with prompt file

```bash
python src/evaluate.py --prompt_file prompts/example_prompts.txt --model gpt-3.5 --api
```

###  2. Run training / marginalization-based prompt generation

```bash
python src/train.py --data data/sample_superglue.json --model gpt-4 --api
```

> NOTE: Set your OpenAI API key via `export OPENAI_API_KEY=...` or directly in code.

---

##  Evaluation Metrics

| Metric | Description |
|--------|-------------|
| EM     | Exact Match |
| LC     | Logical Consistency |
| RD     | Reasoning Depth |
| CS     | Confidence Score |
| IT     | Inference Time |
| Switch | Answer Change Rate |

---

##  Reproducibility

- Random seed = `42`
- Results averaged over 3 runs
- Dataset splits included in `data/`
- Prompt format is documented in `prompts/`

---

##  License

This repository is released under the MIT License.

---

##  Contact

Questions, feedback or collaboration inquiries?  
Feel free to reach out:  
📧 `s.tavangari@alumni.ubc.ca`
