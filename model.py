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

# Step 2 - flatten_patches
def flatten_patches(patches):
    # TODO: flatten each patch's channel and spatial dims into one vector, keep (B, N) leading dims.
    B, N, C, P, P = patches.shape
    return patches.reshape(B, N, -1)

# Step 3 - linear_projection
import torch

def linear_projection(x, weight, bias):
    """Apply y = x @ weight.T + bias with arbitrary leading dims on x."""
    # TODO: compute the affine map y = x @ weight.T + bias
    return x @ weight.T + bias

# Step 4 - project_patches_to_embeddings
import torch

def project_patches_to_embeddings(flat_patches, patch_proj_weight, patch_proj_bias):
    # TODO: Linearly project flattened image patches into the ViT embedding dimension.
    return linear_projection(flat_patches, patch_proj_weight, patch_proj_bias)

# Step 5 - prepend_class_token
import torch

def prepend_class_token(patch_embeddings, class_token):
    """Prepend a learnable [CLS] token to the patch embedding sequence.

    patch_embeddings: (B, num_patches, embed_dim)
    class_token:      (1, 1, embed_dim)
    returns:          (B, num_patches+1, embed_dim)
    """
    # TODO: prepend the [CLS] token to every sequence in the batch
    B, num_patches, embed_dim = patch_embeddings.shape
    class_token = class_token.expand(B, 1, embed_dim)
    return torch.cat((class_token, patch_embeddings), dim=1)

# Step 6 - add_position_embeddings
import torch

def add_position_embeddings(tokens, position_embeddings):
    """Add learnable position embeddings to a (B, S, D) token sequence."""
    # TODO: combine tokens (B, S, D) with position_embeddings (1, S, D) via broadcasting.
    return tokens + position_embeddings

# Step 7 - compute_attention_scores
import torch

def compute_attention_scores(q, k):
    """Compute raw attention scores Q @ K^T.

    q: (..., Sq, d_head)
    k: (..., Sk, d_head)
    returns: (..., Sq, Sk)
    """
    # TODO: compute the raw attention scores as Q times K-transpose
    
    return torch.matmul(q, k.mT)

# Step 8 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_head):
    """Scale raw attention scores so softmax inputs stay well-conditioned."""
    # TODO: Divide raw attention scores by a constant derived from d_head.
    return scores / math.sqrt(d_head)

# Step 9 - apply_attention_mask
def apply_attention_mask(scores, mask):
    # TODO: add an additive mask (0 = allowed, -inf = blocked) to attention scores.
    if mask is None:
        return scores
    return scores + mask

# Step 10 - attention_softmax
import torch

def attention_softmax(masked_scores):
    """Softmax over the last (key) axis of attention scores."""
    # TODO: convert masked attention scores into normalized weights over the key axis
    return torch.softmax(masked_scores, dim=-1)

# Step 11 - attention_context
import torch

def attention_context(attn_weights, v):
    """Combine attention weights with values to produce context vectors."""
    # TODO: return a tensor of shape (..., Sq, d_head) from attn_weights and v
    return attn_weights @ v

# Step 12 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    """Compose score, scale, mask, softmax, and context into full attention."""
    # TODO: compose the five attention primitives into a single forward pass.
    scores = compute_attention_scores(q, k)
    scores = scale_attention_scores(scores, q.shape[-1])
    masked_scores = apply_attention_mask(scores, mask)
    attn_weights = attention_softmax(masked_scores)
    return attention_context(attn_weights, v)

# Step 13 - split_into_heads
import torch

def split_into_heads(x, num_heads):
    """Reshape (B, S, d_model) into (B, num_heads, S, d_head)."""
    # TODO: split the last dim into (num_heads, d_head) and move heads next to batch
    B, S, d_model = x.shape
    d_head = d_model // num_heads
    x = x.reshape(B, S, num_heads, d_head)
    return x.transpose(2, 1)

# Step 14 - merge_heads
import torch

def merge_heads(x):
    """Merge (B, num_heads, S, d_head) back to (B, S, num_heads*d_head)."""
    # TODO: merge the multi-head dimension back into the model dimension
    B, num_heads, S, d_head = x.shape
    trans = x.transpose(1, 2).contiguous()
    return trans.reshape(B, S, -1)

# Step 15 - project_qkv
def project_qkv(x, wq, bq, wk, bk, wv, bv):
    # TODO: project x into separate query, key, and value tensors using three linear layers.
    proj_q = linear_projection(x, wq, bq)
    proj_k = linear_projection(x, wk, bk)
    proj_v = linear_projection(x, wv, bv)

    return proj_q, proj_k, proj_v

# Step 16 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: reshape q, k, v from (B, S, d_model) into (B, num_heads, S, d_head) each
    q_h = split_into_heads(q, num_heads)
    q_k = split_into_heads(k, num_heads)
    q_v = split_into_heads(v, num_heads)

    return q_h, q_k, q_v

# Step 17 - multi_head_attention_scores
import torch

def multi_head_attention_scores(q_h, k_h, v_h, mask=None):
    """Run scaled dot-product attention in parallel across all heads.

    q_h, k_h, v_h: (B, num_heads, S, d_head)
    mask: broadcastable to (B, num_heads, S, S) or None
    returns: (B, num_heads, S, d_head)
    """
    # TODO: run scaled dot-product attention across the head axis
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

