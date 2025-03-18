import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import streamlit as st


st.title("Lab4: Introduction to Diffusion Models")
st.header("1. Load CIFAR-10 Data")
st.write("We use the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Display 5 random images with larger size
indices = np.random.choice(X_train.shape[0], 3, replace=False)
cols = st.columns(3)
for i, col in enumerate(cols):
    sample_image = X_train[indices[i]]
    col.image(sample_image, caption=f'rand_sample_{i+1}', channels='RGB', use_container_width=True)

# Configure GPU
st.header("2. Configure GPU (If Available)")
st.write("Ensure that TensorFlow is set to use GPU resources efficiently.")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        st.write("GPU memory growth configured.")
    except RuntimeError as e:
        st.error(e)
else:
    st.write("No GPU found, using CPU.")

# Function to add noise
def add_noise(images, noise_factor=0.5):
    noise = np.random.normal(0, noise_factor, images.shape)
    return np.clip(images + noise, 0, 1)

train_noisy = add_noise(X_train)
test_noisy = add_noise(X_test)

# Define a simple U-Net model
def build_simple_unet(input_shape=(32, 32, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    outputs = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)
    model = models.Model(inputs, outputs)
    return model

# Define a complex U-Net model
def build_complex_unet(input_shape=(32, 32, 3)):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    
    u1 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    
    u2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same')(c5)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Button to start training
if st.button('Train Models'):
    # Train the simple U-Net model
    with st.spinner('Training Simple U-Net Model...'):
        model_simple = build_simple_unet()
        model_simple.compile(optimizer="adam", loss="mse")
        model_simple.fit(train_noisy[:5000], X_train[:5000],
                            epochs=5, batch_size=32,
                            validation_data=(test_noisy[:1000], X_test[:1000]))

    # Predict using the simple model
    predictions_simple = model_simple.predict(test_noisy[:10])

    # Train the complex U-Net model
    with st.spinner('Training Complex U-Net Model...'):
        model_complex = build_complex_unet()
        model_complex.compile(optimizer="adam", loss="mse")
        model_complex.fit(train_noisy[:5000], X_train[:5000],
                            epochs=5, batch_size=32,
                            validation_data=(test_noisy[:1000], X_test[:1000]))

    # Predict using the complex model
    predictions_complex = model_complex.predict(test_noisy[:10])

    # Display results side by side
    st.header("Results Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Simple U-Net Results")
        fig, axes = plt.subplots(3, 10, figsize=(15, 5))
        for i in range(10):
            axes[0, i].imshow(test_noisy[i])
            axes[0, i].axis("off")
            axes[1, i].imshow(predictions_simple[i])
            axes[1, i].axis("off")
            axes[2, i].imshow(X_test[i])
            axes[2, i].axis("off")
        axes[0, 0].set_ylabel("Noisy", fontsize=10)
        axes[1, 0].set_ylabel("Denoised", fontsize=10)
        axes[2, 0].set_ylabel("Original", fontsize=10)
        st.pyplot(fig)

    with col2:
        st.subheader("Complex U-Net Results")
        fig, axes = plt.subplots(3, 10, figsize=(15, 5))
        for i in range(10):
            axes[0, i].imshow(test_noisy[i])
            axes[0, i].axis("off")
            axes[1, i].imshow(predictions_complex[i])
            axes[1, i].axis("off")
            axes[2, i].imshow(X_test[i])
            axes[2, i].axis("off")
        axes[0, 0].set_ylabel("Noisy", fontsize=10)
        axes[1, 0].set_ylabel("Denoised", fontsize=10)
        axes[2, 0].set_ylabel("Original", fontsize=10)
        st.pyplot(fig)
