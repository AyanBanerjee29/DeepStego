import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class StegoDataset(Dataset):
    def __init__(self, cover_dir, secret_dir, transform_cover=None, transform_secret=None):
        self.cover_dir = cover_dir
        self.secret_dir = secret_dir
        self.cover_images = os.listdir(cover_dir)
        self.secret_images = os.listdir(secret_dir)
        self.transform_cover = transform_cover
        self.transform_secret = transform_secret

    def __len__(self):
        return min(len(self.cover_images), len(self.secret_images))

    def __getitem__(self, idx):
        cover_path = os.path.join(self.cover_dir, self.cover_images[idx])
        secret_path = os.path.join(self.secret_dir, self.secret_images[idx])

        # Cover = RGB, Secret = Grayscale
        try:
            cover = Image.open(cover_path).convert("RGB")
            secret = Image.open(secret_path).convert("L")
        except:
            # Fallback if a file is broken, just reuse the first one
            cover = Image.open(os.path.join(self.cover_dir, self.cover_images[0])).convert("RGB")
            secret = Image.open(os.path.join(self.secret_dir, self.secret_images[0])).convert("L")

        if self.transform_cover:
            cover = self.transform_cover(cover)
        if self.transform_secret:
            secret = self.transform_secret(secret)

        return cover, secret

def get_transforms():
    # --- FIX: REMOVED NORMALIZATION ---
    # We now operate strictly in [0, 1] range. 
    # This prevents the "Dark Image" bug where negative values were clipped.
    
    transform_cover = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(), # Converts 0-255 to 0.0-1.0
    ])

    transform_secret = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(), # Converts 0-255 to 0.0-1.0
    ])
    
    return transform_cover, transform_secret
