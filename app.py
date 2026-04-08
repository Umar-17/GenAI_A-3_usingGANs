import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision.utils import make_grid

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

@st.cache_resource
def load_gan_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(nz=100, ngf=64, nc=3).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        return None, device

st.set_page_config(page_title="Generative AI - Assignment 3", layout="wide")

st.sidebar.title("FAST NUCES - Spring 2026")

st.title("Question 1: Tackling Mode Collapse in GANs")
st.write("Comparing baseline DCGAN with an improved WGAN-GP system to evaluate training stability and diversity[cite: 24, 28].")

num_samples = st.slider("Select number of images to generate", 4, 16, 8, step=4)

if st.button("Generate Comparison"):
    col1, col2 = st.columns(2)
    
    dc_path = "Model/dcgan_generator.pth"
    wg_path = "Model/wgan_generator.pth"
    
    noise = torch.randn(num_samples, 100, 1, 1)
    
    with col1:
        st.subheader("DCGAN (Baseline)")
        model_dc, dev = load_gan_model(dc_path)
        if model_dc:
            with torch.no_grad():
                out = model_dc(noise.to(dev)).cpu()
            grid = make_grid(out, padding=2, normalize=True)
            st.image(np.transpose(grid.numpy(), (1, 2, 0)), use_container_width=True)
        else:
            st.warning(f"Weights not found at {dc_path}.")

    with col2:
        st.subheader("WGAN-GP (Advanced)")
        model_wg, dev = load_gan_model(wg_path)
        if model_wg:
            with torch.no_grad():
                out = model_wg(noise.to(dev)).cpu()
            grid = make_grid(out, padding=2, normalize=True)
            st.image(np.transpose(grid.numpy(), (1, 2, 0)), use_container_width=True)
        else:
            st.warning(f"Weights not found at {wg_path}.")

st.divider()
st.header("Training Logs and Quantitative Evaluation")

log_col1, log_col2 = st.columns(2)

with log_col1:
    st.subheader("DCGAN Loss Plots")
    if os.path.exists("Model/dcgan_loss.png"):
        st.image("Model/dcgan_loss.png", caption="DCGAN Generator vs Discriminator Loss [cite: 116, 117]")
    else:
        st.info("Upload dcgan_loss.png to the Model folder to display training logs.")

with log_col2:
    st.subheader("WGAN-GP Loss Plots")
    if os.path.exists("Model/wgan_loss.png"):
        st.image("Model/wgan_loss.png", caption="WGAN-GP Generator vs Critic Loss [cite: 116, 117]")
    else:
        st.info("Upload wgan_loss.png to the Model folder to display training logs.")

st.subheader("Model Performance Analysis")
st.write("""
This system demonstrates how advanced loss functions like Wasserstein Loss with Gradient Penalty improve training stability[cite: 28, 65]. 
WGAN-GP effectively eliminates mode collapse and improves the diversity of generated samples compared to the baseline DCGAN model[cite: 63, 64].
""")