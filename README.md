Markdown
# E-CR-JEPAv2: Enhanced Cross-Modal Joint-Embedding Predictive Architecture

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)

E-CR-JEPAv2 is a robust, multi-modal representation learning framework designed to fuse Synthetic Aperture Radar (SAR) and Optical continuous spectrum imagery. Built and optimized for the **SEN12MS-CR-TS** dataset, this network leverages a Joint-Embedding Predictive Architecture (JEPA) to align disparate spatial modalities without representation collapse.

## 🧠 Core Architecture

Unlike standard contrastive learning methods that struggle with the massive semantic gap between radar scattering backwash and RGB features, E-CR-JEPAv2 utilizes an asymmetric student-teacher predictive structure.

* **SAR Encoder (Student):** A custom `ViTEncoderTrunk` embedded with spatial 2D-ALiBi matrices to capture scale-invariant radar geometries.
* **Optical Target Encoder (Teacher):** A pre-trained Hugging Face **DINOv2** backbone (`dinov2-small`). DINOv2 natively enforces high feature variance and provides a highly mature, structured optical latent space for the SAR features to map into.
* **Predictor (`DeeperXALiBiPredictor`):** An enhanced cross-attention block with a hidden MLP bottleneck layer designed to translate masked SAR tokens into the target DINOv2 semantic space.
* **Decoupled Retrieval Heads:** Separates common multi-modal features from unique single-modal identifiers for downstream retrieval tasks.

## ⚡ Key Optimizations & Stability Guardrails

This model includes custom numerical and architectural optimizations to ensure stable training over extended epochs (e.g., 100+ epochs):

1. **Target Normalization (PSA LayerNorm):** Applies `LayerNorm` to both the predictor outputs and the DINOv2 targets prior to MSE calculation, eliminating variance scale mismatches and accelerating Predictive Spatial Alignment (PSA) convergence.
2. **NaN-Resistant Training Loop:** Features an automated gradient integrity guard that skips corrupt optimization steps and aborts gracefully if numeric instability (NaN/Inf) is detected, preserving the last stable checkpoint.
3. **Variance Clamping (DeCUR & SIGReg):** Replaces standard epsilon division with absolute variance clamping (`torch.clamp(var, min=1e-3)`) to prevent standard deviation explosions under Automatic Mixed Precision (AMP) Float16.
4. **Optimized Masking Ratio:** Utilizes a 50% asymmetric masking ratio, providing the SAR predictor with sufficient structural context to bridge the modality gap to DINOv2.

## 📂 Dataset Specification

The model is trained on the **SEN12MS-CR-TS** dataset.
* **Total Paired Samples:** 7,680 multi-modal patches.
* **Data Allocation:** 85% Training (6,528 steps) | 15% Validation (1,152 steps).
* **Expected Directory:** Mounted at `/kaggle/input/sen12ms-cr-ts` or equivalent root structure.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* PyTorch 2.0+ (CUDA enabled)
* Transformers (Hugging Face)
* Kaggle Environments (Optional, but recommended for automated dataset handling)

### Execution Structure

Ensure your dataset is properly mounted, then execute the main training pipeline. (If running via a server script rather than a notebook environment, trigger the orchestration file):

```bash
python server.py
The training process is divided into two automated phases:

Phase A: 100-Epoch Pre-training (Logging loss, PSA, and DeCUR metrics).

Phase B: Leak-Free Information Retrieval Evaluation on the validation split.

Final metrics are automatically written to /kaggle/working/final_evaluation_metrics.json.

📊 Training Performance
By isolating the modalities and mapping SAR directly to DINOv2, the model achieves highly stable latent alignment:

Final Overall Loss: ~0.1604

Final PSA Loss: ~0.2106

Final DeCUR Loss: ~0.1183
(Metrics recorded over a 100-epoch continuous run)

Author: Vaatsal
