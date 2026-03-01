import time
import numpy as np
import random
import cv2
import glob
import os
import argparse
import polySim as pm
from numba import njit
from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage.filters import laplace, threshold_otsu
from scipy import ndimage

"""
DENOISING DATA GENERATION SCRIPT
Author: Nikhil Chaurasia
Methodology: Grain-wise Superimposition of Real Microstructural Noise
Task: Generating Otsu-Thresholded Noisy Input vs. Clean Template Ground Truth
"""

def microsimulator(size=512):
    """Generates a clean polycrystalline template using polySim."""
    nucleation_rate = random.randint(20, 40)
    growth_rate = random.randint(4, 5)
    img = np.zeros((size, size), dtype=np.uint16)
    pm.generate_structure(img, nucleation_rate, growth_rate)
    return img

@njit
def phasepick(micro, label, n, size):
    """Identifies the bounding box of a specific grain 'n'."""
    thresh = 255
    a = b = c = d = 0
    x1 = x2 = y3 = y4 = 0
    
    # Identify grain area and top-most x (x1)
    for x in range(size[0]):
        for y in range(size[1]):
            if micro[x][y] == n:
                a += 1
                if a == 1: x1 = x
                label[x][y] = thresh
    
    # Bottom-most x (x2)
    for x in range(size[0]-1, -1, -1):
        for y in range(size[1]-1, -1, -1):
            if label[x][y] == thresh:
                b += 1
                if b == 1: x2 = x
    
    # Right-most y (y3)
    for y in range(size[1]-1, -1, -1):
        for x in range(size[0]-1, -1, -1):
            if label[x][y] == thresh:
                c += 1
                if c == 1: y3 = y

    # Left-most y (y4)
    for y in range(size[1]):
        for x in range(size[0]):
            if label[x][y] == thresh:
                d += 1
                if d == 1: y4 = y
                
    window_size = (abs(y4-y3)+1, abs(x1-x2)+1)
    return x1, x2, y3, y4, window_size, label

@njit
def window_merge_loop(x1, x2, y3, y4, label, p, image_copy):
    """Pastes the noise crop 'p' into the designated grain area."""
    for x in range(x1, x2 + 1):
        for y in range(y4, y3 + 1):
            if label[x][y] == 255:
                # Map the noise crop coordinates to the grain coordinates
                image_copy[x][y] = p[abs(x1-x)][abs(y4-y)]
    return image_copy

def window_merge(sizes, image_copy, x1, x2, y3, y4, label, noise_crops):
    """Prepares and rotates a random real noise crop to fill a grain."""
    im_idx = random.randint(0, len(noise_crops)-1)
    theta = random.randint(0, 180)
    
    p = cv2.imread(noise_crops[im_idx], 0)
    p = ndimage.rotate(p, theta, reshape=False, mode='reflect')
    p = resize(p, (sizes, sizes))
    p = img_as_ubyte(p)
    
    return window_merge_loop(x1, x2, y3, y4, label, p, image_copy)

def generating_boundary(actual_noisy, micro_template, prob):
    """
    Blends noisy grain textures into the boundary network and returns 
    the noisy image and the clean polycrystalline template (Ground Truth).
    """
    size = actual_noisy.shape
    normalised_img = (np.rint((actual_noisy-actual_noisy.min())*(255/(actual_noisy.max()-actual_noisy.min())))).astype(np.uint8)
    avg = int(np.mean(normalised_img))
    
    lis = [(255-avg), random.randint(40, 50)]
    m = random.choices(lis, weights=[prob, (1-prob)])[0]

    # Extract clean boundaries for Ground Truth and blending
    trace = laplace(micro_template, ksize=3)
    trace[trace != 0] = 255
    trace_dil = cv2.dilate(trace.astype(np.uint8), np.ones((3,3), np.uint8))
    
    # This is the clean polycrystalline template (Ground Truth)
    # Saved as black boundaries on white background
    clean_template = 255 - trace_dil
    
    # Add gaussian blur to simulate experimental imaging effects
    blur = cv2.GaussianBlur(clean_template, (5,5), 0)
    
    # Create the noisy microstructure by blending grain noise into the boundary gaps
    for i in range(size[0]):
        for j in range(size[1]):
            if blur[i][j] > m:
                blur[i][j] = normalised_img[i][j]
    
    return blur, clean_template

def main():
    parser = argparse.ArgumentParser(description="Otsu-Thresholded Denoising Dataset Generator")
    parser.add_argument("--noise_dir", type=str, help="Path to cropped real noise images")
    parser.add_argument("--output", type=str, default="./denoising_dataset")
    parser.add_argument("--count", type=int, default=10)
    
    try:
        args = parser.parse_args()
    except:
        args, _ = parser.parse_known_args()

    if not args.noise_dir:
        print("\n[!] No noise directory provided via command line.")
        args.noise_dir = input("Please enter the path to your noise crops folder: ").strip().replace('"', '').replace("'", "")
        
    if not os.path.exists(args.noise_dir):
        print(f"Error: Folder not found at {args.noise_dir}")
        return

    noise_crops = glob.glob(os.path.join(args.noise_dir, "*.png"))
    if not noise_crops:
        print(f"Error: No .png files found in {args.noise_dir}")
        return

    # Define the two primary folder names for the final paired dataset
    os.makedirs(os.path.join(args.output, "noisy_input"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "ground_truth_template"), exist_ok=True)

    print(f"\nStarting generation of {args.count} image pairs...")
    for k in range(args.count):
        micro = microsimulator()
        size = micro.shape
        actual_noisy = np.zeros(size, dtype=np.uint8)
        
        unique_grains = np.unique(micro)
        for n in unique_grains:
            label = np.zeros(size, dtype=np.uint8)
            x1, x2, y3, y4, q_size, label = phasepick(micro, label, n, size)
            q_max = np.amax(q_size)
            actual_noisy = window_merge(q_max, actual_noisy, x1, x2, y3, y4, label, noise_crops)

        # 1. Generate the noisy simulated micrograph and the clean GT template
        noisy_micrograph, clean_gt = generating_boundary(actual_noisy, micro, 0.8)
        
        # 2. Apply Otsu Thresholding to the noisy micrograph to create the "noisy input"
        try:
            val = threshold_otsu(noisy_micrograph)
            # Use thresholding to create the binary input for the model
            noisy_input_binary = (noisy_micrograph > val).astype(np.uint8) * 255
        except ValueError:
            # Fallback for uniform images
            noisy_input_binary = np.zeros_like(noisy_micrograph)
        
        # 3. Save the Otsu version (Noisy Input) and the Clean Template (Ground Truth)
        cv2.imwrite(os.path.join(args.output, f"noisy_input/{k}.png"), noisy_input_binary)
        cv2.imwrite(os.path.join(args.output, f"ground_truth_template/{k}.png"), clean_gt)
        
        if (k + 1) % 5 == 0 or (k + 1) == args.count:
            print(f"Progress: {k+1}/{args.count} pairs generated.")

if __name__ == "__main__":
    main()