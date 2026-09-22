import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'flower_hand.jpeg'
img_bgr = cv2.imread(img_path)

if img_bgr is None:
    raise FileNotFoundError(f"Could not load image at '{img_path}'. Check the filename/path.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
height, width = img_gray.shape

print("--- 1. MATRIX REPRESENTATION ---")
print(f"Color Matrix Shape (H x W x C): {img_rgb.shape}")
print(f"Grayscale Matrix Shape (H x W): {img_gray.shape}")

print(f"\nGrayscale Min Value: {img_gray.min()}")
print(f"Grayscale Max Value: {img_gray.max()}")

print(f"RGB Overall Min Value: {img_rgb.min()}")
print(f"RGB Overall Max Value: {img_rgb.max()}")

min_per_channel = img_rgb.min(axis=(0, 1))
max_per_channel = img_rgb.max(axis=(0, 1))
print(f"RGB Min per Channel (R, G, B): {min_per_channel}")
print(f"RGB Max per Channel (R, G, B): {max_per_channel}")

print("\nSample 5x5 intensity sub-matrix from Grayscale Image:")
print(img_gray[100:105, 100:105])

sampled_low = cv2.resize(img_gray, (width // 8, height // 8), interpolation=cv2.INTER_NEAREST)
sampled_reconstructed = cv2.resize(sampled_low, (width, height), interpolation=cv2.INTER_NEAREST)

def quantize_image(img, bits):
    levels = 2**bits
    factor = 256 // levels
    return (img // factor) * factor

quant_4bit = quantize_image(img_gray, 4)
quant_2bit = quantize_image(img_gray, 2)

res_50pct = cv2.resize(img_rgb, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
res_25pct = cv2.resize(img_rgb, (width // 4, height // 4), interpolation=cv2.INTER_AREA)

plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title(f"Original ({width}x{height})")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(sampled_reconstructed, cmap='gray')
plt.title("Sampled Down (1/8th Spatial)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(quant_4bit, cmap='gray')
plt.title("Quantized (4-Bit / 16 Levels)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(quant_2bit, cmap='gray')
plt.title("Quantized (2-Bit / 4 Levels)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(res_50pct)
plt.title(f"Resolution 50% ({width//2}x{height//2})")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(res_25pct)
plt.title(f"Resolution 25% ({width//4}x{height//4})")
plt.axis("off")

plt.tight_layout()
plt.show()