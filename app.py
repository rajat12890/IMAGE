import streamlit as st
import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import os

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Face Real ESRGAN Upscaler")

# Title and Description
st.title("Face Real ESRGAN UpScale: 2x 4x 8x")
st.markdown("""
This is an unofficial demo for Real-ESRGAN. Scales the resolution of a photo. 
This model shows better results on faces compared to the original version.
<br>
Telegram BOT: <a href="https://t.me/restoration_photo_bot" target="_blank">https://t.me/restoration_photo_bot</a>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center;'>
Twitter <a href='https://twitter.com/DoEvent' target='_blank'>Max Skobeev</a> | 
<a href='https://huggingface.co/sberbank-ai/Real-ESRGAN' target='_blank'>Model card</a>
</div>
""", unsafe_allow_html=True)

# Cache models to avoid reloading on every rerun
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    st.info(f"Loading models on: {device}")

    # Ensure weights directory exists
    os.makedirs('weights', exist_ok=True)

    model2 = RealESRGAN(device, scale=2)
    model2.load_weights('weights/RealESRGAN_x2.pth', download=True)
    
    model4 = RealESRGAN(device, scale=4)
    model4.load_weights('weights/RealESRGAN_x4.pth', download=True)
    
    model8 = RealESRGAN(device, scale=8)
    model8.load_weights('weights/RealESRGAN_x8.pth', download=True)
    
    st.success("Models loaded successfully!")
    return model2, model4, model8, device

model2, model4, model8, device = load_models()

def inference(image, size, current_model2, current_model4, current_model8, current_device):
    if image is None:
        st.error("Please upload an image.")
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
            if width >= 5000 or height >= 5000:
                st.error("The image is too large for 8x upscaling. Please try a smaller image or a lower upscale factor.")
                return None
            model = current_model8
            
        result = model.predict(image.convert('RGB'))
        st.info(f"Image upscaled with {size} model on {current_device}.")
        
    except torch.cuda.OutOfMemoryError as e:
        st.error(f"CUDA Out of Memory Error: {e}. Please try a smaller image or a lower upscale factor.")
        # Attempt to reload the model if OOM occurs. This might not always solve for persistent OOM.
        # For Streamlit, consider adding st.rerun() if you want to force a re-initialization of the cached resource.
        # However, for OOM, it's often better to just inform the user.
        if size == '2x':
            global model2 # Access global variable if not passed as current_model
            model2 = RealESRGAN(current_device, scale=2)
            model2.load_weights('weights/RealESRGAN_x2.pth', download=False)
        elif size == '4x':
            global model4
            model4 = RealESRGAN(current_device, scale=4)
            model4.load_weights('weights/RealESRGAN_x4.pth', download=False)
        else:
            global model8
            model8 = RealESRGAN(current_device, scale=8)
            model8.load_weights('weights/RealESRGAN_x8.pth', download=False)
        
    except Exception as e:
        st.error(f"An error occurred during inference: {e}")
        
    return result

# Streamlit UI elements
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    
    input_image = None
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Image", use_column_width=True)

    st.subheader("Upscale Options")
    selected_size = st.radio(
        "Select Resolution Model:",
        ("2x", "4x", "8x"),
        index=0, # Default to 2x
        horizontal=True
    )

    if st.button("Upscale Image"):
        if input_image is not None:
            with st.spinner("Upscaling image... This might take a moment."):
                output_image = inference(input_image, selected_size, model2, model4, model8, device)
            
            if output_image is not None:
                st.session_state['output_image'] = output_image
        else:
            st.error("Please upload an image before upscaling.")

with col2:
    st.subheader("Output Image")
    if 'output_image' in st.session_state and st.session_state['output_image'] is not None:
        st.image(st.session_state['output_image'], caption="Upscaled Image", use_column_width=True)
    else:
        st.write("Upscaled image will appear here.")
        
    # Optional: Download button for the output image
    if 'output_image' in st.session_state and st.session_state['output_image'] is not None:
        st.download_button(
            label="Download Upscaled Image",
            data=st.session_state['output_image'].tobytes(),
            file_name=f"upscaled_{uploaded_file.name if uploaded_file else 'image'}_{selected_size}.png",
            mime="image/png"
        )