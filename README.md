# MIMIC-CXR-VQA — Official Code Repository

This repository contains the **official implementation** of the creation code of the dataset MIMIC-CXR-VQA.

📄 **Paper:** *MIMIC-CXR-VQA: A Medical Visual Question Answering Dataset Constructed with LLaMA-based Annotations*


🔗 **Link:** [Comming Soon!!]

-----

## Dataset

![Alt text](datasetimg.png)

## Baselines

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate swinbed
```

-----


## 2\. Running Experiments 🚀


### Base Training Pipeline

To execute the full training pipeline, follow these sequential steps:

1.  **Change Directory:** First, navigate to the training directory:
    ```bash
    cd train
    ```

2.  **Stage 1: NLL Training (Frozen Encoder)**
    Execute the initial Negative Log-Likelihood (NLL) training run with the model's encoder frozen:
    ```bash
    ./nll_train_freeze_econder.sh
    ```

3.  **Stage 2: NLL Training (Unfrozen Encoder)**
    Follow up with the NLL training using an unfrozen encoder for full model fine-tuning:
    ```bash
    ./nll_train_unfreeze_econder.sh
    ```

4.  **Stage 3: Reinforcement Learning (RL) Training**
    Complete the process by executing the Reinforcement Learning (RL) training phase:
    ```bash
    ./rl_train.sh
    ```