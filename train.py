import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.networks import StegoEncoder, StegoDecoder
from utils.loaders import StegoDataset, get_transforms
from utils.encryption import Scrambler
import os

# --- Advanced Config ---
EPOCHS = 200            # ResU-Net converges faster than standard U-Net
BATCH_SIZE = 2          # Adjusted for GPU memory
LR = 0.0001             # Slow and steady
COVER_WEIGHT = 20.0     # High priority on invisibility (L1 Loss)
SECRET_WEIGHT = 10.0     # Standard priority on Data (BCE Loss)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    os.makedirs("saved_models", exist_ok=True)
    
    # 1. Get Split Transforms
    t_cover, t_secret = get_transforms()
    
    dataset = StegoDataset("data/train_covers", "data/train_secrets", t_cover, t_secret)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder = StegoEncoder().to(DEVICE)
    decoder = StegoDecoder().to(DEVICE)
    scrambler = Scrambler(size=512) # Initialize Encryption

    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    
    # --- HYBRID LOSS FUNCTIONS ---
    criterion_cover = nn.L1Loss()           # From Model 1 (Sharpness)
    criterion_secret = nn.BCEWithLogitsLoss() # From Model 2 (Binary Accuracy)

    print(f"🚀 Starting Hybrid Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        total_loss = 0
        
        for covers, secrets in dataloader:
            covers, secrets = covers.to(DEVICE), secrets.to(DEVICE)

            # Scramble Secret (Encryption)
            scrambled_secrets = scrambler.scramble(secrets).to(DEVICE)

            optimizer.zero_grad()

            # 1. Forward Pass
            stego_images = encoder(covers, scrambled_secrets)
            recovered_scrambled = decoder(stego_images)

            # 2. Calculate Hybrid Loss
            # Compare Stego vs Cover using L1 (Sharpness)
            loss_cover = criterion_cover(stego_images, covers)
            
            # Compare Recovered vs Scrambled using BCE (Bits)
            loss_secret = criterion_secret(recovered_scrambled, scrambled_secrets)
            
            loss = (COVER_WEIGHT * loss_cover) + (SECRET_WEIGHT * loss_secret)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Total Loss: {total_loss:.4f} | Cover L1: {loss_cover.item():.4f}")

    # Save
    torch.save(encoder.state_dict(), "saved_models/encoder.pth")
    torch.save(decoder.state_dict(), "saved_models/decoder.pth")
    print("✅ Hybrid Model Saved!")

if __name__ == "__main__":
    train()
