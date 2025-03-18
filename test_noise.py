import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import streamlit as st
from PIL import Image

def add_gaussian_noise(images, mean=0.0, std=0.1):
    images_tensor = torch.from_numpy(images).float() / 255.0
    noise = torch.randn_like(images_tensor) * std + mean
    noised_image = images_tensor + noise
    noised_image = torch.clamp(noised_image, 0.0, 1.0)
    noised_image = (noised_image * 255).byte().numpy()
    return noised_image

def plot_images_and_histograms(images, titles):
    num_images = len(images)
    num_channels = 3

    fig, axes = plt.subplots(num_images, num_channels + 1, figsize=(20, 5 * num_images))
    
    if num_images == 1:
        axes = [axes]

    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(title)
        axes[i, 0].axis('off')

        for j, (channel, color) in enumerate(zip(cv2.split(image), ('b', 'g', 'r'))):
            histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
            axes[i, j + 1].plot(histogram, color=color)
            axes[i, j + 1].set_xlim([0, 256])
            axes[i, j + 1].set_title(f'{title} - {color.upper()} Channel')
            axes[i, j + 1].set_xlabel('Pixel Value')
            axes[i, j + 1].set_ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(fig)

st.title("Image Noising")
st.sidebar.header("Noising Parameters")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

num_iterations = st.sidebar.slider("Number of Iterations", 1, 10, 5)
mean = st.sidebar.slider("Mean", -1.0, 1.0, 0.0)
stddev = st.sidebar.slider("Standard Deviation", 0.0, 1.0, 0.1)

start = st.button("Apply Noise")
if start and uploaded_file is not None:

    image = np.array(Image.open(uploaded_file))

    current_image = image.copy()
    images = [image.copy()]
    titles = ['Original']

    for i in range(num_iterations):
        current_image = add_gaussian_noise(current_image, mean, stddev)
        images.append(current_image)
        titles.append(f'Noised {i+1}')

    plot_images_and_histograms(images, titles)