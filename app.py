import streamlit as st
import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import os

# Set page configuration for a wider layout
st.set_page_config(layout="centered", page_title="Image Upscaler Powered by Real-ESRGAN")

# --- Title and Description ---
st.title("Image Upscaler: Powered by Real-ESRGAN")
st.markdown("""
Enhance your images with our advanced upscaler! This tool uses a specialized version of Real-ESRGAN,
optimized for better results, especially on faces. Choose your desired upscale factor (2x, 4x, or 8x)
and transform your low-resolution images into high-quality visuals.
""")


# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    st.info(f"Loading models on: **{device.type.upper()}**")

    # Ensure weights directory exists
    os.makedirs('weights', exist_ok=True)

    model2 = RealESRGAN(device, scale=2)
    model2.load_weights('weights/RealESRGAN_x2.pth', download=True)
    
    model4 = RealESRGAN(device, scale=4)
    model4.load_weights('weights/RealESRGAN_x4.pth', download=True)
    
    model8 = RealESRGAN(device, scale=8)
    model8.load_weights('weights/RealESRGAN_x8.pth', download=True)
    
    st.success("All models loaded successfully!")
    return model2, model4, model8, device

model2, model4, model8, device = load_models()

# --- Inference Function ---
def inference(image, size, current_model2, current_model4, current_model8, current_device):
    if image is None:
        st.error("Please upload an image to upscale.")
        return None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    result = None
    try:
        if size == '2x':
            model = current_model2
        elif size == '4x':
            model = current_model4
        else: # size == '8x'
            width, height = image.size
            # Add a more generous size check for 8x to prevent immediate OOM
            if width * height > 4000 * 4000 and current_device.type == 'cpu': # Arbitrary limit for CPU
                 st.error("The image might be too large for 8x upscaling on CPU. Consider a smaller image or a lower upscale factor.")
                 return None
            elif width * height > 2000 * 2000 and current_device.type == 'cuda': # Arbitrary limit for GPU
                 st.warning("For 8x upscaling, very large images can consume significant GPU memory. Proceeding with caution.")
            
            model = current_model8
            
        result = model.predict(image.convert('RGB'))
        st.info(f"Image successfully upscaled using the **{size}** model.")
        
    except torch.cuda.OutOfMemoryError as e:
        st.error(f"**CUDA Out of Memory Error:** Your image is too large for the selected upscale factor on this GPU. Please try a smaller image or a lower upscale factor (e.g., 2x or 4x).")
        # In case of OOM, we don't attempt to reload as it's often a user input issue rather than a model issue.
        # Informing the user is more effective.
    except Exception as e:
        st.error(f"An unexpected error occurred during upscaling: {e}. Please try again or with a different image.")
        
    return result

# --- Streamlit UI Layout ---
st.write("---") # Horizontal line for visual separation

col1, col2 = st.columns([1, 1]) # Adjust column ratio if needed

with col1:
    st.subheader("Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
    
    input_image = None
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Image", use_column_width=True)
    else:
        st.info("Upload an image (PNG, JPG, JPEG, WEBP) to get started!")

    st.subheader("Select Upscale Factor")
    selected_size = st.radio(
        "Choose resolution:",
        ("2x", "4x", "8x"),
        index=0, # Default to 2x
        horizontal=True,
        help="Select how many times you want to increase the image resolution."
    )

    st.markdown("---") # Another horizontal line
    if st.button("✨ Upscale Image", use_container_width=True, type="primary"):
        if input_image is not None:
            with st.spinner("Processing image... This may take a moment depending on image size and selected upscale factor."):
                output_image = inference(input_image, selected_size, model2, model4, model8, device)
            
            if output_image is not None:
                st.session_state['output_image'] = output_image
        else:
            st.error("Please upload an image before clicking 'Upscale Image'.")

with col2:
    st.subheader("Upscaled Image")
    if 'output_image' in st.session_state and st.session_state['output_image'] is not None:
        st.image(st.session_state['output_image'], caption="Upscaled Image", use_column_width=True)
        
        st.markdown("---") # Horizontal line before download button
        # Optional: Download button for the output image
        st.download_button(
            label="Download Upscaled Image",
            data=st.session_state['output_image'].tobytes(),
            file_name=f"upscaled_{uploaded_file.name if uploaded_file else 'image'}_{selected_size}.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.info("Your upscaled image will appear here after processing.")

st.write("---")

st.markdown("""
<div style='text-align: center; margin-top: 20px;'>
Developed by <strong>Rajat Parihar</strong>
</div>
""", unsafe_allow_html=True)
