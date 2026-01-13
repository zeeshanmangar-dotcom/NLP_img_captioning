import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Image Captioning App",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_model():
    """
    Load the pre-trained image captioning model, processor, and tokenizer.
    This function is cached to avoid reloading the model on every interaction.
    """
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, processor, tokenizer, device

def generate_caption(image, model, processor, tokenizer, device):
    """
    Generate a caption for the given image using the loaded model.
    
    Args:
        image: PIL Image object
        model: Pre-trained vision-encoder-decoder model
        processor: Image processor
        tokenizer: Text tokenizer
        device: torch device (CPU or GPU)
    
    Returns:
        str: Generated caption
    """
    # Preprocess the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=16,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode the generated tokens to text
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return caption

def load_image_from_url(url):
    """
    Download and load an image from a URL.
    
    Args:
        url: Image URL string
    
    Returns:
        PIL.Image: Loaded image or None if error
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image.convert("RGB")
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")
        return None

# Main app
def main():
    st.title("🖼️ Image Captioning App")
    st.write("Generate captions for your images using AI!")
    
    # Load model with progress indicator
    with st.spinner("Loading AI model... (This may take a moment on first run)"):
        model, processor, tokenizer, device = load_model()
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "🔗 Image URL", "📷 Webcam"])
    
    image = None
    
    # Tab 1: File Upload
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "gif"],
            help="Upload an image from your computer"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Tab 2: URL Input
    with tab2:
        image_url = st.text_input(
            "Enter image URL",
            placeholder="https://example.com/image.jpg",
            help="Paste a direct link to an image"
        )
        if image_url:
            with st.spinner("Loading image from URL..."):
                image = load_image_from_url(image_url)
                if image is not None:
                    st.image(image, caption="Image from URL", use_container_width=True)
    
    # Tab 3: Webcam Capture
    with tab3:
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image = Image.open(camera_photo).convert("RGB")
            st.image(image, caption="Captured Photo", use_container_width=True)
    
    # Generate caption button
    st.write("---")
    
    if image is not None:
        if st.button("🎯 Generate Caption", type="primary", use_container_width=True):
            with st.spinner("Generating caption..."):
                try:
                    caption = generate_caption(image, model, processor, tokenizer, device)
                    st.success("Caption generated successfully!")
                    st.write("### 📝 Generated Caption:")
                    st.info(caption)
                except Exception as e:
                    st.error(f"Error generating caption: {str(e)}")
    else:
        st.warning("⚠️ Please select an image using one of the methods above.")
    
    # Footer
    st.write("---")
    st.caption("Powered by Hugging Face Transformers | Model: nlpconnect/vit-gpt2-image-captioning")

if __name__ == "__main__":
    main()
