import torch
from PIL import Image
from models.networks import StegoDecoder
from utils.loaders import get_transforms
from utils.encryption import Scrambler  # <--- Added this
import torchvision.utils as vutils
import sys
import os

def reveal_secret(stego_path, output_path="recovered_secret.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Components
    # We use the same Scrambler (Key) used in training
    scrambler = Scrambler(size=512) 
    decoder = StegoDecoder().to(device)
    
    # 2. Load Weights
    if not os.path.exists("saved_models/decoder.pth"):
        print("Error: 'saved_models/decoder.pth' not found. Train the model first!")
        return

    decoder.load_state_dict(torch.load("saved_models/decoder.pth", map_location=device))
    decoder.eval()

    # 3. Prepare Image
    # We use the 'cover' transform because the stego image looks like a cover
    t_cover, _ = get_transforms()
    
    try:
        stego_image = Image.open(stego_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: File '{stego_path}' not found.")
        return

    stego_tensor = t_cover(stego_image).unsqueeze(0).to(device)

    # 4. Reveal & Descramble
    print(f"Analyzing {stego_path}...")
    with torch.no_grad():
        # Step A: Extract the hidden signal (which is still scrambled noise)
        raw_output = decoder(stego_tensor)
        
        # Step B: Convert Logits to Probabilities (Sigmoid)
        # In the new model, the decoder outputs raw scores, so we must apply Sigmoid
        probs = torch.sigmoid(raw_output)
        
        # Step C: Decrypt (Descramble) to get the actual QR code
        recovered_qr = scrambler.descramble(probs)

    # 5. Save
    vutils.save_image(recovered_qr, output_path, normalize=True)
    print(f"Success! Secret revealed and saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reveal.py <path_to_stego_image>")
    else:
        in_file = sys.argv[1]
        out_file = sys.argv[2] if len(sys.argv) > 2 else "recovered_secret.png"
        reveal_secret(in_file, out_file)
