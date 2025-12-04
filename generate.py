import qrcode
import os

def generate_qrs(count=100, output_dir="data/train_secrets"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {count} QR codes...")

    for i in range(1, count + 1):
        # We use High Error Correction (H) so the QR survives 
        # being hidden inside another image.
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )

        # The data is just a dummy string
        data = f"Hidden_Secret_Data_{i}"
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img.save(os.path.join(output_dir, f"qr_{i}.png"))

    print(f"Done! Check {output_dir}")

if __name__ == "__main__":
    generate_qrs()
