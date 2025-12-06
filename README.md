# MIMIC-CXR-VQA — Official Code Repository

This repository contains the **official implementation** of the creation code of the dataset MIMIC-CXR-VQA.

📄 **Paper:** *MIMIC-CXR-VQA: A Medical Visual Question Answering Dataset Constructed with LLaMA-based Annotations*


🔗 **Link:** [Comming Soon!!]

-----

## Dataset

![Alt text](datasetimg.png)

The MIMIC-CXR-VQA dataset contains millions of question–answer pairs automatically generated from the radiology reports of MIMIC-CXR. Each sample includes an image, a clinically relevant question, and a concise answer produced through template-guided generation to ensure semantic consistency. The goal is to provide a large-scale resource for training and evaluating medical Visual Question Answering models.

## 🤖 Baselines

- **Baseline 1:** [SwinVED-SCST](https://github.com/Maa1604/SwinVED-SCST)
- **Baseline 2:** [BLIP-2 MultiView](https://github.com/Maa1604/BLIP-2-MultiView)

These models serve as initial reference points for benchmarking VQA performance on the dataset. Each baseline represents a different approach: one uses an encoder–decoder framework with self-critical sequence training, while the other leverages multi-view reasoning to enhance visual understanding.