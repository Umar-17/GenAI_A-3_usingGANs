# Generative AI Assignment 3: GAN Explorer

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License" />
</div>

<br/>

## 🎯 Objective
This repository contains the implementation of three distinct tasks focusing on Generative Adversarial Networks (GANs) and image-to-image translation models. The integrated Streamlit application allows users to interact with and explore the deployed models.

### Question 1: Mode Collapse (DCGAN vs WGAN-GP)
- **Objective:** Address the common problem of **mode collapse** by improving training stability.
- **Implementation:** Compares a baseline **Deep Convolutional GAN (DCGAN)** with an improved **Wasserstein GAN with Gradient Penalty (WGAN-GP)**.
- **Dataset:** Anime Faces / Pokemon Sprites.

### Question 2: Pix2Pix (Sketch to Photo Translation)
- **Objective:** Implement a conditional GAN for paired image-to-image translation.
- **Implementation:** Uses a **U-Net** Generator and a PatchGAN Discriminator to translate sketches into fully colored photos.
- **Features:** Interactive drawing canvas in the app to translate free-hand sketches.

### Question 3: CycleGAN (Unpaired Translation)
- **Objective:** Perform unpaired image translation where aligned dataset pairs are not available.
- **Implementation:** Uses a **ResNet (6-block)** Generator and Cycle-Consistency Loss to translate images between two distinct domains (e.g., Photo ↔ Sketch).
- **Features:** Supports translation in both directions using uploaded images or the interactive drawing canvas.

---

## 🛠️ Environment Setup
- **Frameworks:** PyTorch, Torchvision
- **Deployment Platform:** Streamlit
- **Accelerator:** GPU T4 x2 (Dual GPU) utilized for accelerated training on Kaggle.

---

## 🧠 Model Architectures

1. **DCGAN / WGAN-GP:** Transposed Convolutional Generator with Discriminator/Critic architectures.
2. **Pix2Pix:** U-Net based Generator with skip connections to preserve spatial details.
3. **CycleGAN:** ResNet based Generator (6 residual blocks) for robust domain translation without structure loss.

---

## 🚀 Repository Content

- **`Notebook/`**: Complete Jupyter/Kaggle notebooks containing the training code for DCGAN, WGAN-GP, Pix2Pix, and CycleGAN.
- **`Model/`**: Contains the saved PyTorch model weights for all three questions:
  - `dcgan_generator_final.pt`
  - `wgangp_generator_final.pt`
  - `pix2pix_export_q2.pt`
  - `cyclegan_weights.pt`
- **`models.py`**: Model architecture definitions (DCGAN, WGAN-GP, U-Net, ResNet).
- **`app.py`**: The interactive **Streamlit** deployed application demonstrating the answers to all three questions.

---

## 🖼️ Application Usage
Run the following command to start the Streamlit application:
```bash
streamlit run app.py
```
Use the sidebar to navigate between Q1 (Noise to Image), Q2 (Sketch to Photo), and Q3 (CycleGAN translation).

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
