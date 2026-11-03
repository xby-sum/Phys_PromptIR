# Phys-PromptIR: Physics-Inspired Prompt Learning for Limited-Data Image Restoration

Official PyTorch implementation of the paper: **"Phys-PromptIR: Physics-Inspired Prompt Learning with Adaptive Residual Modulation for Limited-Data Image Restoration"**.

> **[Baoyue Xu]**, [Yinuo Jiang], [Yanyan Zhang]
> [Paper Link / arXiv Link (If available)] | [Project Page (Optional)]

## Overview

All-in-one image restoration must accommodate heterogeneous degradations and distribution shifts between synthetic and real images. **Phys-PromptIR** introduces a physics-inspired prompt learning framework for limited-data image restoration. It features:
- **Physical Prior Estimator (PPE)**: Produces bounded attenuation-like, global atmospheric-light, and additive-degradation representations.
- **Physical Prompt Fusion (PPF)**: Maps complementary representations into the encoder feature space.
- **Prompt Reliability Estimator (PRE)**: An input-dependent gate that controls the residual contribution of physical prompts without replacing the visual feature pathway.
- **Two-Stage Fine-tuning Strategy**: Supports efficient target-domain adaptation using only 100 paired images per task.

## Environment Preparation

We recommend using Conda to manage the environment.

```bash
# Clone the repository
git clone https://github.com/xby-sum/Phys_PromptIR.git
cd Phys-PromptIR

# Create a new conda environment
conda create -n physpromptir python=3.8
conda activate physpromptir

# Install dependencies
pip install -r requirements.txt