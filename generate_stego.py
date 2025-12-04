import torch
from PIL import Image
from models.networks import StegoEncoder
# We do NOT import StegoDecoder here
from utils.loaders import get_transforms
from utils.encryption import Scrambler
import torchvision.utils as vutils
import sys
import os

def generate_stego(cover_path, secret_path, output_name="result_stego.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_cover, t_secret = get_transforms()

    # 1. Load ONLY the Encoder
    encoder = StegoEncoder().to(device)
    
    if not os.path.exists("saved_models/encoder.pth"):
        print("Error: 'saved_models/encoder.pth' not found.")
        return

    # Load weights
    encoder.load_state_dict(torch.load("saved_models/encoder.pth", map_location=device))
    encoder.eval() 

    # 2. Initialize Scrambler (Must match training size!)
    scrambler = Scrambler(size=512)

    # 3. Load Images
    try:
        cover = Image.open(cover_path).convert("RGB")
        secret = Image.open(secret_path).convert("L")
    except FileNotFoundError:
        print("Error: Input images not found.")
        return
    
    # Transform
    cover_tensor = t_cover(cover).unsqueeze(0).to(device)
    secret_tensor = t_secret(secret).unsqueeze(0).to(device)

    print(f"Encoding {secret_path} into {cover_path}...")

    with torch.no_grad():
        # A. Scramble the Secret
        scr_secret = scrambler.scramble(secret_tensor).to(device)
        
        # B. Encode (Hide)
        stego_image = encoder(cover_tensor, scr_secret)

    # 4. Save Result
    vutils.save_image(stego_image, output_name)
    
    print(f"Success! Stego image saved to: {output_name}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_stego.py <cover_image> <qr_code> [optional_output_filename]")
    else:
        cover = sys.argv[1]
        secret = sys.argv[2]
        output = sys.argv[3] if len(sys.argv) > 3 else "result_stego.png"
        
        generate_stego(cover, secret, output)