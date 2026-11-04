"""
Vision-Language Model from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - split_image_into_patches
def split_image_into_patches(image, patch_size):
    B, C, H, W = image.shape

    if patch_size <= 0 or H % patch_size or W % patch_size:
        raise ValueError("Invalid patch_size for image dimensions")

    patches = image.unfold(2, patch_size, patch_size) \
                   .unfold(3, patch_size, patch_size)

    return patches.permute(0, 2, 3, 1, 4, 5) \
                  .reshape(B, -1, C, patch_size, patch_size)

