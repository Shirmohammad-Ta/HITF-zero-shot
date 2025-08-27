# Hybrid Instruction Tuning Framework (HITF) for Zero-Shot Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![academia](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://www.academia.edu/143573481/Hybrid_Instruction_Tuning_with_Marginalization_for_Zero_Shot_Reasoning_in_Language_Models)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of the paper:  
**"Hybrid Instruction Tuning with Marginalization for Zero-Shot Reasoning in Language Models"**

This repository contains the code and resources for the **Hybrid Instruction Tuning Framework (HITF)**, a novel method that enhances the zero-shot reasoning capabilities of Large Language Models (LLMs) by dynamically combining human-annotated and self-generated examples through a task-aware mixture selector.

---

## 🔍 Overview

Large language models often struggle with complex, multi-step reasoning tasks. HITF addresses this by:

-   **Dynamic Example Selection**: A lightweight selector module computes an optimal blend of high-quality human data and diverse self-generated examples for each input task.
-   **Intermediate Answer Marginalization**: Improves robustness and logical consistency by probabilistically combining multiple reasoning paths.
-   **Adaptive Prompt Construction**: Builds few-shot prompts tailored to the specific task, improving generalization without model retraining.
-   **Efficient Inference**: Achieves significant performance gains with minimal increase in computational cost.

<div align="center">
<img src="assets/architecture.png" alt="HITF Architecture" width="600"/>
</div>

---

## ✨ Key Features

-   **Zero-Shot Reasoning Improvement**: Enhances accuracy, coherence, and depth of reasoning in models like GPT-3.5, GPT-4, and T5.
-   **Plug-and-Play Framework**: Can be applied on top of existing pre-trained models without full fine-tuning.
-   **Resource Efficient**: Uses a lightweight selector network, making it suitable for low-resource environments.
-   **Comprehensive Evaluation**: Tested on SuperGLUE, MMLU, TriviaQA, and custom arithmetic/logical reasoning benchmarks.

---

## 📈 Results

Our method outperforms strong baselines across multiple metrics:

| Model/Method       | Exact Match (EM) ↑ | Logical Consistency (LC) ↑ | Reasoning Depth (RD) ↑ | Inference Time (IT) (ms) ↓ |
|--------------------|-------------------|---------------------------|----------------------|--------------------------|
| Standard Fine-Tuning (FT)   | 65.7              | 70.2                      | 3.4                  | 120                      |
| Self-Generated Data (SGDT)  | 58.9              | 64.5                      | 2.9                  | 110                      |
| Fixed Hybrid (HFT)          | 70.2              | 75.1                      | 3.9                  | 125                      |
| GPT-3.5            | 67.3              | 71.8                      | 3.5                  | 150                      |
| GPT-4              | 73.6              | 78.9                      | 4.1                  | 180                      |
| **HITF (Ours)**    | **76.5**          | **82.7**                  | **4.5**              | **115**                  |

---

## 🚀 Quick Start

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Shirmohammad-Ta/HITF-zero-shot.git
    cd HITF-zero-shot
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Prepare Datasets**: Place your `D_human` and `D_self` datasets in the `data/` directory. Format: JSONL files with `{"instruction": ..., "reasoning": ..., "answer": ...}`.

2.  **Encode Tasks & Train Selector**:
    ```python
    from src.encoder import TaskEncoder
    from src.selector import DynamicSelector

    encoder = TaskEncoder()
    selector = DynamicSelector()
    selector.train(encoder, data_path="data/")
    ```

3.  **Run Inference**:
    ```python
    from src.inference import HITFInference

    hitf = HITFInference(selector_model_path="models/selector.pt")
    query = "What is the square root of 144?"
    result = hitf.query(query, k=5)  # k = support set size
    print(result['answer'])
    ```

---

##  📜 License
This project is licensed under the MIT License - see the LICENSE file for details.


## 📞 Contact
- **Author:** Shirmohammad Tavangari  
- **Email:** s.tavangari@alumni.ubc.ca  
- **Institution:** University of British Columbia, Canada
- **Here is the link to the paper:** [Hybrid Instruction Tuning with Marginalization for Zero-Shot Reasoning in Language Models (Academia.edu)](https://www.academia.edu/143573481/Hybrid_Instruction_Tuning_with_Marginalization_for_Zero_Shot_Reasoning_in_Language_Models)

