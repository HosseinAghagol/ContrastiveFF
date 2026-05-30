# Contrastive Forward-Forward: A Training Algorithm of Vision Transformer

*   **Journal Version:** [ScienceDirect - Neural Networks (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0893608025007476)
*   **Preprint Version:** [arXiv:2502.00571](https://arxiv.org/abs/2502.00571)

This repository contains the official implementation of **Contrastive Forward-Forward (CFF)**, a novel training algorithm for Vision Transformers (ViT) that serves as a biologically plausible alternative to backpropagation.

## 📌 Overview
Backpropagation (BP) is the standard for training neural networks but lacks biological plausibility due to its global nature and memory requirements. The **Forward-Forward (FF)** algorithm was recently introduced as a brain-inspired alternative, but it often suffers from a performance gap compared to BP.

**Contrastive Forward-Forward (CFF)** bridges this gap by combining the layer-wise independence of the FF algorithm with the powerful representation-learning insights of **Supervised Contrastive Learning**.

### Key Features
*   **Biologically Plausible:** Utilizes local loss functions after each layer, eliminating the need for a global backward pass.
*   **Superior Performance:** Outperforms baseline FF by up to **10% in accuracy** and accelerates convergence speed by **5 to 20 times**.
*   **Vision Transformer Integration:** The first study to successfully extend the FF framework to modern **Vision Transformer (ViT)** architectures.
*   **Marginal Contrastive Loss:** Introduces a modified contrastive loss that uses a margin parameter ($m$) to gradually refine representations across layers.
*   **Parallel Training:** Supports a unique **pipeline-parallel** architecture that allows different layers to be trained simultaneously on multiple GPUs, significantly reducing training time per epoch.

---

## 🚀 How It Works
The CFF training process is divided into two main stages:

1.  **Stage 1: Layer-wise Encoder Training**
    *   The data undergoes two paths of random augmentation.
    *   Each layer is updated immediately after its forward pass using the **Marginal Contrastive Loss**, which encourages samples of the same class to be closer in the representation space while distancing different classes.
    *   Layers are trained independently and can be processed in parallel.

2.  **Stage 2: Linear Classifier Training**
    *   With the encoder weights frozen, a simple MLP head is trained using standard Cross-Entropy (CE) to map the learned representations to correct labels.

---

## 📊 Experimental Results
CFF has been evaluated on four benchmark datasets: **MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet**. 

*   **Robustness:** CFF demonstrates higher robustness to **overfitting** as model complexity increases compared to BP.
*   **Inaccurate Supervision:** In scenarios with noisy labels (e.g., 20% incorrect labels), CFF exhibits less performance degradation than backpropagation.
*   **Efficiency:** Achieve competitive accuracy to BP while enabling more efficient GPU utilization through simultaneous **Data-Parallel and Pipeline-Parallel** training.

---

## 📑 Citation
If you find this work useful for your research, please cite the original paper:

```bibtex
@article{aghagolzadeh2025contrastive,
  title={Contrastive Forward-Forward: A training algorithm of vision transformer},
  author={Aghagolzadeh, Hossein and Ezoji, Mehdi},
  journal={Neural Networks},
  volume={192},
  pages={107867},
  year={2025},
  publisher={Elsevier}
}
```

---

## 📧 Contact
**Hossein Aghagolzadeh** - hossein.aghagol@gmail.com, ho.golzadeh@stu.nit.ac.ir  
**Mehdi Ezoji** - m.ezoji@nit.ac.ir  
*Faculty of Electrical and Computer Engineering, Babol Noshirvani University of Technology, Babol, Iran*
