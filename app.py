import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import os

# Import your project modules
from models.networks import StegoEncoder, StegoDecoder
from utils.encryption import Scrambler

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "saved_models"

st.set_page_config(page_title="Invisible QR Steganography", layout="centered")

# --- Helper Functions ---

@st.cache_resource
def load_models():
    """
    Loads models once and caches them in memory to speed up the app.
    """
    # Initialize Models
    encoder = StegoEncoder().to(DEVICE)
    decoder = StegoDecoder().to(DEVICE)
    scrambler = Scrambler(size=512)  # Size must match training

    # Check paths
    enc_path = os.path.join(MODEL_DIR, "encoder.pth")
    dec_path = os.path.join(MODEL_DIR, "decoder.pth")

    if not os.path.exists(enc_path) or not os.path.exists(dec_path):
        st.error(f"❌ Models not found in {MODEL_DIR}. Please train the model first.")
        return None, None, None

    # Load Weights
    encoder.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE))
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, scrambler

def transform_image(image, is_secret=False):
    """
    Preprocesses the uploaded image to 512x512 tensor.
    Matches logic in utils/loaders.py
    """
    transform_list = [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
    t = transforms.Compose(transform_list)
    
    # Handle Grayscale vs RGB
    if is_secret:
        image = image.convert("L") # QR code is grayscale
    else:
        image = image.convert("RGB") # Cover is RGB
        
    return t(image).unsqueeze(0).to(DEVICE)

def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor back to a PIL Image for display/download.
    """
    image = tensor.squeeze(0).cpu().detach()
    image = torch.clamp(image, 0, 1) # Ensure valid pixel range
    return transforms.ToPILImage()(image)

def convert_to_bytes(pil_image):
    """
    Converts PIL image to bytes for download button.
    """
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# --- Main UI ---

st.title("🕵️ Invisible QR Steganography")
st.markdown("Hide a QR Code inside a high-quality image using **Deep Learning (ResNet + U-Net)**.")

# Load models immediately
encoder, decoder, scrambler = load_models()

if encoder:
    tab1, tab2 = st.tabs(["🔒 Hide QR (Encode)", "🔓 Reveal QR (Decode)"])

    # ==========================
    # TAB 1: ENCODE (HIDE)
    # ==========================
    with tab1:
        st.header("Create a Stego Image")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Upload Cover Image")
            cover_file = st.file_uploader("Choose a photo (JPG/PNG)", type=["jpg", "png", "jpeg"], key="cover")
            if cover_file:
                cover_img = Image.open(cover_file)
                # FIX: use_container_width instead of use_column_width
                st.image(cover_img, caption="Cover Image", use_container_width=True)

        with col2:
            st.subheader("2. Upload QR Code")
            secret_file = st.file_uploader("Choose a QR code (PNG)", type=["png", "jpg"], key="secret")
            if secret_file:
                secret_img = Image.open(secret_file)
                # FIX: use_container_width instead of use_column_width
                st.image(secret_img, caption="Secret QR", use_container_width=True)

        # Process
        if cover_file and secret_file:
            if st.button("🚀 Generate Invisible QR", type="primary"):
                with st.spinner("Encrypting and fusing images..."):
                    # 1. Transform
                    cover_tensor = transform_image(cover_img, is_secret=False)
                    secret_tensor = transform_image(secret_img, is_secret=True)

                    with torch.no_grad():
                        # 2. Scramble
                        scrambled_secret = scrambler.scramble(secret_tensor).to(DEVICE)
                        
                        # 3. Encode
                        stego_tensor = encoder(cover_tensor, scrambled_secret)
                        
                        # 4. Post-process
                        result_image = tensor_to_image(stego_tensor)

                    st.success("Steganography Complete!")
                    
                    st.divider()
                    st.subheader("Result")
                    # FIX: use_container_width instead of use_column_width
                    st.image(result_image, caption="Stego Image (Contains Hidden Data)", use_container_width=True)
                    
                    # Download Button
                    btn = st.download_button(
                        label="📥 Download Stego Image",
                        data=convert_to_bytes(result_image),
                        file_name="stego_result.png",
                        mime="image/png"
                    )

    # ==========================
    # TAB 2: DECODE (REVEAL)
    # ==========================
    with tab2:
        st.header("Reveal Hidden Data")
        st.write("Upload an image generated by this tool to recover the hidden QR code.")
        
        stego_file = st.file_uploader("Upload Stego Image", type=["png"], key="decode")
        
        if stego_file:
            stego_img = Image.open(stego_file)
            st.image(stego_img, caption="Uploaded Stego Image", width=300)
            
            if st.button("🔍 Reveal Secret"):
                with st.spinner("Scanning and Decrypting..."):
                    # 1. Transform
                    stego_tensor = transform_image(stego_img, is_secret=False)
                    
                    with torch.no_grad():
                        # 2. Decode (Extract Raw Noise)
                        raw_output = decoder(stego_tensor)
                        
                        # 3. Sigmoid (Logits -> Probabilities)
                        probs = torch.sigmoid(raw_output)
                        
                        # 4. Descramble (Decrypt)
                        recovered_tensor = scrambler.descramble(probs)
                        
                        # 5. Convert to PIL
                        recovered_image = tensor_to_image(recovered_tensor)
                    
                    st.success("Secret Found!")
                    st.image(recovered_image, caption="Recovered QR Code", width=300)
                    
                    st.download_button(
                        label="📥 Download QR Code",
                        data=convert_to_bytes(recovered_image),
                        file_name="recovered_secret.png",
                        mime="image/png"
                    )