# Tackling Mode Collapse in Generative Adversarial Networks (GANs)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License" />
</div>

<br/>

## 🎯 Objective
The objective of this project is to design and implement a Generative Adversarial Network (GAN) system and address the common problem of **mode collapse** by improving training stability using advanced techniques. The implementation includes:
- A baseline **Deep Convolutional GAN (DCGAN)**
- An improved **Wasserstein GAN with Gradient Penalty (WGAN-GP)**

The system demonstrates how advanced loss functions improve the training stability and the diversity of generated images.

---

## 🛠️ Environment Setup
- **Platform:** Kaggle
- **Accelerator:** GPU T4 x2 (Dual GPU) utilized for accelerated training.
- **Datasets Used:**
  - [Pokemon Sprites](https://www.kaggle.com/datasets/jackemartin/pokemon-sprites)
  - [Anime Faces (64×64)](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)

---

## 🧠 Model Architecture

### 1. Baseline Model: DCGAN Configuration
- **Input Noise Vector (z):** 100-dimensional
- **Image Size:** 64 × 64
- **Generator:** Transposed Convolution Layers, Batch Normalization, ReLU Activation, Output Activation: Tanh.
- **Discriminator:** Convolutional Layers, LeakyReLU Activation, Output Activation: Sigmoid.
- **Goals:** Generate realistic images from noise, learn data distribution of the training dataset, and provide baseline performance for comparison.

### 2. Advanced Model: WGAN-GP Configuration
- **Discriminator Replacement:** Replaced with a **Critic** (no sigmoid activation).
- **Loss Function:** Wasserstein Loss.
- **Gradient Penalty:** λ = 10.
- **Critic updates per Generator update:** 5.
- **Goals:** Eliminate mode collapse, improve diversity of generated samples, and ensure stable training dynamics.

---

## 🚀 Implementation Details

### Part 1: Data Preparation
1. Load dataset (Pokemon / Anime Faces).
2. Resize all images to `64 × 64`.
3. Normalize images to range `[-1, 1]`.
4. Create PyTorch `DataLoader`.

### Part 2: Forward Pass & Loss Computation
**DCGAN:**
1. Sample random noise vector (z).
2. Generate fake image using the Generator.
3. Pass real and fake images to Discriminator.
4. Compute **Binary Cross Entropy (BCE) Loss**.

**WGAN-GP:**
1. Generate fake images.
2. Compute critic scores for real and fake images.
3. Calculate **Wasserstein loss**.
4. Apply **Gradient Penalty**.

### Part 3: Training Setup
- **Optimizer:** `Adam` (Learning Rate: `0.0002`, Betas: `(0.5, 0.999)`)
- **Training Strategy:** DCGAN was trained first to establish a baseline, followed by WGAN-GP for comparative performance.
- **Training Techniques:** To fit the Kaggle T4×2 environment, techniques like Mixed Precision (`torch.cuda.amp`), batch size adjustments (e.g., 64), and frequent checkpointing were used.

---

## 🖼️ Visualization and Deliverables

A comprehensive visualization module has been implemented to display generated images from both DCGAN and WGAN-GP, allowing for a direct comparison of diversity and quality.

### Repository Content
- **`Notebook/gen-ai-a-3-q1.ipynb`**: Complete PyTorch implementation of DCGAN and WGAN-GP, including dataset preparation, model architecture, training loop, and evaluation charts (Generator vs. Discriminator/Critic Loss).
- **`Model/`**: Contains the saved PyTorch model weights.
  - `dcgan_generator.pth`
  - `wgan_generator.pth`
- **`app.py`**: A **Streamlit** deployed application demonstrating both models interactively, enabling real-world testing and side-by-side comparison.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
