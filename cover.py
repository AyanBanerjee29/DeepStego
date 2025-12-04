import requests
import os
import time

def download_images(category, count, output_dir):
    """
    Downloads random images from LoremFlickr based on a keyword.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Downloading {count} '{category}' images to {output_dir} ---")
    
    for i in range(count):
        # URL structure: https://loremflickr.com/{width}/{height}/{keyword}
        # We add a random parameter at the end to ensure we don't get cached same images
        url = f"https://loremflickr.com/256/256/{category}?random={i}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                file_path = os.path.join(output_dir, f"{category}_{i+1}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"[{i+1}/{count}] Saved {file_path}")
            else:
                print(f"[{i+1}/{count}] Failed (Status {response.status_code})")
        except Exception as e:
            print(f"[{i+1}/{count}] Error: {e}")
        
        # Be polite to the server, wait a bit between requests
        time.sleep(1)

if __name__ == "__main__":
    # 1. Download Nature images for training
    download_images("nature", 50, "data/train_covers")
    
    # 2. Download City images for training (adds variety)
    download_images("city", 50, "data/train_covers")
    
    # 3. Download one specific test image
    download_images("cat", 1, "data/test")
    
    print("\nDone! Your data folders are ready.")
