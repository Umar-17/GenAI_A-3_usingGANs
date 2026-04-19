import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import OrderedDict
from streamlit_drawable_canvas import st_canvas

# Import our architectures from models.py
from models import DCGAN_Generator, WGAN_Generator, UNetGenerator, ResNetGenerator

st.set_page_config(page_title="GAN Explorer | FAST-NU", layout="wide")
st.title("Generative AI Assignment 3: GAN Explorer")
st.sidebar.title("Navigation")

# Helper function to convert output tensors to displayable images
def tensor_to_image(tensor):
    image = tensor.cpu().detach().squeeze(0)
    image = (image + 1) / 2.0  # Un-normalize from [-1, 1] to [0, 1]
    image = transforms.ToPILImage()(image)
    return image

# Navigation
question = st.sidebar.radio("Select Question:", [
    "Q1: Mode Collapse (DCGAN vs WGAN)",
    "Q2: Pix2Pix (Sketch to Photo)",
    "Q3: CycleGAN (Unpaired Translation)"
])

# =======================================================
# Q1: DCGAN vs WGAN
# =======================================================
if question == "Q1: Mode Collapse (DCGAN vs WGAN)":
    st.header("Noise to Image Generation")
    st.markdown("Compare the outputs of standard DCGAN against the improved WGAN-GP.")
    
    model_choice = st.radio("Select Model:", ("DCGAN", "WGAN-GP"))
    
    if st.button("Generate Images"):
        with st.spinner(f"Generating with {model_choice}..."):
            try:
                # Load correct model
                if model_choice == "DCGAN":
                    model = DCGAN_Generator(nz=100, ngf=64, nc=3)
                    model.load_state_dict(torch.load("Model/dcgan_generator_final.pt", map_location="cpu"))
                else:
                    model = WGAN_Generator(nz=100, ngf=64, nc=3)
                    model.load_state_dict(torch.load("Model/wgangp_generator_final.pt", map_location="cpu"))
                
                model.eval()
                
                # Generate 4 random images
                noise = torch.randn(4, 100, 1, 1)
                with torch.no_grad():
                    fakes = model(noise)
                
                # Display images in a grid
                cols = st.columns(4)
                for i in range(4):
                    with cols[i]:
                        st.image(tensor_to_image(fakes[i].unsqueeze(0)), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading model: {e}")

# =======================================================
# Q2: Pix2Pix 
# =======================================================
elif question == "Q2: Pix2Pix (Sketch to Photo)":
    st.header("Paired Sketch-to-Photo Translation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Draw a Sketch")
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=2,
            stroke_color="black",
            background_color="white",
            height=256,
            width=256,
            drawing_mode="freedraw",
            key="canvas_q2",
        )
    
    if st.button("Translate to Photo"):
        if canvas_result.image_data is not None:
            with st.spinner("Translating..."):
                try:
                    # Convert Canvas to PIL Image
                    input_img = Image.fromarray((canvas_result.image_data).astype(np.uint8)).convert('RGB')
                    
                    # Transform
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                    input_tensor = transform(input_img).unsqueeze(0)
                    
                    # Load Model
                    model = UNetGenerator(in_channels=3, out_channels=3)
                    checkpoint_q2 = torch.load("Model/pix2pix_export_q2.pt", map_location="cpu")
                    # Extract ONLY the generator weights from the dictionary
                    model.load_state_dict(checkpoint_q2["generator"])
                    model.eval()
                    
                    # Inference
                    with torch.no_grad():
                        output_tensor = model(input_tensor)
                    
                    with col2:
                        st.subheader("Generated Photo")
                        st.image(tensor_to_image(output_tensor), use_container_width=True)
                except Exception as e:
                    st.error(f"Translation Error: {e}")

# =======================================================
# Q3: CycleGAN 
# =======================================================
elif question == "Q3: CycleGAN (Unpaired Translation)":
    st.header("Unpaired Image Translation (CycleGAN)")
    
    direction = st.radio("Select Translation Direction:", ("Sketch ➡️ Photo", "Photo ➡️ Sketch"))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image (Upload or Draw)")
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        
        # Fallback to canvas if no upload
        if not uploaded_file:
            st.markdown("*Or draw something below:*")
            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=2,
                stroke_color="black",
                background_color="white",
                height=128,
                width=128,
                drawing_mode="freedraw",
                key="canvas_q3",
            )
            
    if st.button("Translate"):
        with st.spinner("Translating..."):
            try:
                # Prepare Image
                if uploaded_file is not None:
                    input_img = Image.open(uploaded_file).convert('RGB')
                else:
                    input_img = Image.fromarray((canvas_result.image_data).astype(np.uint8)).convert('RGB')

                # Transform (Resizing to 128x128 as per Q3 PDF requirements)
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                input_tensor = transform(input_img).unsqueeze(0)
                
                # Load CycleGAN model from single dictionary file
                model = ResNetGenerator(n_residual_blocks=6)
                checkpoint = torch.load("Model/cyclegan_weights.pt", map_location="cpu")
                
                # Depending on how it was saved, try common key names
                if direction == "Sketch ➡️ Photo":
                    raw_weights = checkpoint.get('G_AB', checkpoint) 
                else:
                    raw_weights = checkpoint.get('G_BA', checkpoint)
                
                # Fix the Multi-GPU "module." prefix issue
                clean_weights = OrderedDict()
                for k, v in raw_weights.items():
                    name = k[7:] if k.startswith('module.') else k # Remove 'module.'
                    clean_weights[name] = v
                    
                model.load_state_dict(clean_weights)
                model.eval()
                
                # Inference
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                
                with col2:
                    st.subheader("Output Translation")
                    st.image(tensor_to_image(output_tensor), use_container_width=True)
            except Exception as e:
                st.error(f"Translation Error: {e}")