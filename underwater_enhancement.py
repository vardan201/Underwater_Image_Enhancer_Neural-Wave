"""
==============================================================================
Underwater Image Enhancement (UIE) - Research Paper Implementation
==============================================================================

Paper: "A robust and efficient non-learning approach for underwater image 
        enhancement using color balancing and multiscale fusion"
Published in: Scientific Reports (2026) 16:3182
DOI: https://doi.org/10.1038/s41598-025-33170-9

Pipeline Overview:
  Step 1: Adaptive Color Balancing (Red/Blue compensation + Gray World)
  Step 2: Input 1 — Morphological Processed Residuals (MPR)
  Step 3: Input 2 — Normalized Unsharp Masking
  Step 4: Weight Maps (Laplacian contrast + Saliency + Saturation)
  Step 5: Multiscale Laplacian Pyramid Fusion

Author: Implementation based on the methodology described in the paper.
==============================================================================
"""

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from skimage.morphology import reconstruction, area_opening
import os
import sys
import time


# =============================================================================
# STEP 1: ADAPTIVE COLOR BALANCING (Equations 11, 12 + Gray World)
# =============================================================================

def adaptive_color_balance(img_bgr):
    """
    Performs adaptive color balancing for underwater images.
    
    Phase 1: Red and Blue channel compensation (Eq. 11, 12)
      - Compensates for wavelength-dependent attenuation in water
      - Red light (long wavelength) is attenuated most, green least
      - Uses green channel as reference for compensation
    
    Phase 2: Gray World normalization
      - Normalizes all channels to have equal mean intensity
      - Removes remaining color casts
    
    Parameters:
        img_bgr: Input image in BGR format (uint8, 0-255)
    
    Returns:
        color_balanced: Color-balanced image (float64, 0-1 range)
    """
    # Normalize to [0, 1]
    img = img_bgr.astype(np.float64) / 255.0
    
    # Extract channels (OpenCV uses BGR order)
    Ib = img[:, :, 0]  # Blue channel
    Ig = img[:, :, 1]  # Green channel
    Ir = img[:, :, 2]  # Red channel
    
    # Compute mean intensities for each channel
    mean_r = np.mean(Ir)
    mean_g = np.mean(Ig)
    mean_b = np.mean(Ib)
    
    # Color compensation coefficient (empirically set to 1.0 as per paper)
    alpha = 1.0
    
    # ----- Red Channel Compensation (Equation 11) -----
    # Irc(x) = Ir(x) + α·(μg - μr)·(1 - Ir(x)·Ig(x))
    # The term (μg - μr) measures attenuation difference
    # The term (1 - Ir(x)·Ig(x)) provides spatial adaptivity:
    #   - More compensation where red is weak (dark areas)
    #   - Less compensation where red is already strong (bright areas)
    Irc = Ir + alpha * (mean_g - mean_r) * (1.0 - Ir * Ig)
    
    # ----- Blue Channel Compensation (Equation 12) -----
    # Ibc(x) = Ib(x) + α·(μg - μb)·(1 - Ib(x)·Ig(x))
    Ibc = Ib + alpha * (mean_g - mean_b) * (1.0 - Ib * Ig)
    
    # Clip to valid range
    Irc = np.clip(Irc, 0.0, 1.0)
    Ibc = np.clip(Ibc, 0.0, 1.0)
    
    # Reconstruct compensated image (BGR order)
    compensated = np.stack([Ibc, Ig, Irc], axis=2)
    
    # ----- Gray World Normalization (Soft Blend) -----
    # Scale each channel so that all channels have the same mean
    # Use soft blending (70/30) to avoid over-desaturation
    # while still removing the color cast
    channel_means = np.mean(compensated, axis=(0, 1))
    overall_mean = np.mean(channel_means)
    
    gray_world = np.zeros_like(compensated)
    for c in range(3):
        if channel_means[c] > 1e-6:
            gray_world[:, :, c] = compensated[:, :, c] * (overall_mean / channel_means[c])
        else:
            gray_world[:, :, c] = compensated[:, :, c]
    gray_world = np.clip(gray_world, 0.0, 1.0)
    
    # Blend: mostly Gray World corrected, but retain some original color vibrancy
    blend_factor = 0.85
    color_balanced = blend_factor * gray_world + (1.0 - blend_factor) * compensated
    color_balanced = np.clip(color_balanced, 0.0, 1.0)
    
    return color_balanced


# =============================================================================
# STEP 2: MORPHOLOGICAL PROCESSED RESIDUALS (Equations 2-10)
# =============================================================================

def morphological_operator_M(I_residual, t, s):
    """
    Morphological operator M as defined in Equations 7-9.
    
    M(I) = R_I(min(I, S_t(I) | {min{I}, max{I}}))
    
    Where:
      - S_t(I) = (I >= t) is the thresholding function (Eq. 8)
      - '|' maps binary 0→min(I), 1→max(I) 
      - min(.) is pointwise minimum
      - R_I(A) is morphological reconstruction of mask I using marker A
    
    The size criterion (Eq. 9) removes regions smaller than s pixels.
    
    Parameters:
        I_residual: Residual component (positive part), float64
        t: Amplitude threshold (default 0.01)
        s: Minimum region size in pixels for area opening
    
    Returns:
        Processed residual with preserved significant edges
    """
    if np.max(I_residual) < 1e-10:
        return np.zeros_like(I_residual)
    
    # ---- Thresholding (Equation 8) ----
    # S_t(I) = (I >= t) — identifies significant residual regions
    binary_mask = (I_residual >= t).astype(np.uint8)
    
    # ---- Size criterion (Equation 9) ----
    # Remove connected components smaller than s pixels (area opening)
    # This eliminates noise while preserving meaningful structures
    if s > 1:
        binary_mask = area_opening(binary_mask, area_threshold=s).astype(np.uint8)
    
    # ---- Construct marker for morphological reconstruction (Equation 7) ----
    # Map binary mask: 0 → min(I_residual), 1 → max(I_residual)
    min_val = np.min(I_residual)
    max_val = np.max(I_residual)
    
    if max_val - min_val < 1e-10:
        return np.zeros_like(I_residual)
    
    mapped = np.where(binary_mask > 0, max_val, min_val)
    
    # Pointwise minimum of original residual and mapped marker
    marker = np.minimum(I_residual, mapped)
    
    # Ensure marker <= mask for reconstruction (required by skimage)
    mask_for_recon = np.maximum(I_residual, marker)
    marker_for_recon = np.minimum(marker, mask_for_recon)
    
    # ---- Morphological reconstruction by dilation (Equation 7) ----
    # Grows the marker under the constraint of the mask image
    # Preserves the shape of significant residual peaks
    try:
        result = reconstruction(marker_for_recon, mask_for_recon, method='dilation')
    except Exception:
        # Fallback: simply mask the residual
        result = I_residual * binary_mask.astype(np.float64)
    
    return result


def morphological_processed_residuals(img, t=0.01, s=3, c=1.0):
    """
    Applies Morphological Processed Residuals (MPR) to enhance edges.
    
    This implements Equations 2-10 from the paper:
      1. Low-pass filter the image (Eq. 2)
      2. Compute residual (Eq. 3)
      3. Split into positive/negative parts (Eq. 4)
      4. Apply morphological operator M to each part (Eq. 6-9)
      5. Reconstruct with contrast control (Eq. 10)
    
    Parameters:
        img: Color-balanced image (float64, 0-1)
        t: Amplitude threshold for significant residuals (0.01)
        s: Minimum region size in pixels (3)
        c: Contrast control coefficient (1.0)
    
    Returns:
        Edge-enhanced image (float64, 0-1)
    """
    result = np.zeros_like(img)
    
    for ch in range(3):
        u = img[:, :, ch].copy()
        
        # ---- Gaussian Low-Pass Filter (Equation 2) ----
        # I = u * L, where L is a Gaussian kernel
        I_lowpass = cv2.GaussianBlur(u, (0, 0), sigmaX=2.0)
        
        # ---- Compute Residual (Equation 3) ----
        # Res(u) = u - I
        Res = u - I_lowpass
        
        # ---- Split into Positive and Negative Parts (Equation 4) ----
        # Ires+ = 0.5 * (Res + |Res|)  (positive part)
        # Ires- = 0.5 * (Res - |Res|)  (negative part)
        Ires_pos = np.maximum(Res, 0.0)
        Ires_neg = np.minimum(Res, 0.0)
        
        # ---- Apply Morphological Operator M (Equations 6-9) ----
        M_pos = morphological_operator_M(Ires_pos, t, s)
        M_neg = morphological_operator_M(np.abs(Ires_neg), t, s)
        
        # ---- Reconstruct with Contrast Control (Equation 10) ----
        # Iout = I + (M(Ires+) - M(Ires-)) · c
        result[:, :, ch] = I_lowpass + (M_pos - M_neg) * c
    
    return np.clip(result, 0.0, 1.0)


# =============================================================================
# STEP 3: NORMALIZED UNSHARP MASKING (Equation 13)
# =============================================================================

def normalized_unsharp_masking(img):
    """
    Creates the second fusion input using Normalized Unsharp Masking.
    
    Equation 13: S = (I + N{I - G*I}) / 2
    
    Where:
      - I is the color-balanced image
      - G*I is the Gaussian-filtered version
      - N{.} is linear normalization (histogram stretching)
      - The result enhances sharpness without parameter tuning
    
    This technique mitigates degradation from light scattering
    while avoiding the parameter sensitivity of traditional unsharp masking.
    
    IMPORTANT: Normalization uses GLOBAL min/max across all channels
    to preserve inter-channel color ratios and prevent desaturation.
    
    Parameters:
        img: Color-balanced image (float64, 0-1)
    
    Returns:
        Sharpened image (float64, 0-1)
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    
    # Compute detail signal: I - G*I
    detail = img - blurred
    
    # Normalize GLOBALLY across all channels (histogram stretching)
    # Using global min/max preserves color ratios between channels
    # Per-channel normalization would make all channels identical = grayscale
    global_min = np.min(detail)
    global_max = np.max(detail)
    
    if global_max - global_min > 1e-10:
        normalized_detail = (detail - global_min) / (global_max - global_min)
    else:
        normalized_detail = np.zeros_like(detail)
    
    # Sharpened image (Equation 13)
    # S = (I + N{I - G*I}) / 2
    S = (img + normalized_detail) / 2.0
    
    return np.clip(S, 0.0, 1.0)


# =============================================================================
# STEP 4: WEIGHT MAP COMPUTATION (Equations 14-15)
# =============================================================================

def compute_laplacian_contrast_weight(img):
    """
    Laplacian contrast weight (WL).
    
    Computes the absolute value of the Laplacian filter applied to
    the luminance channel. Highlights regions with significant
    intensity transitions (edges and textures).
    
    Parameters:
        img: Input image (float64, 0-1, BGR)
    
    Returns:
        WL: Laplacian contrast weight map
    """
    # Compute luminance
    L = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
    
    # Apply Laplacian filter
    L_uint8 = (L * 255.0).astype(np.uint8)
    laplacian = cv2.Laplacian(L_uint8, cv2.CV_64F)
    WL = np.abs(laplacian) / 255.0
    
    return WL


def compute_saliency_weight(img):
    """
    Saliency weight (WS).
    
    Based on the biological principle of center-surround contrast.
    Computed as the absolute difference between the luminance and
    its large-scale Gaussian blur (mean of the image).
    Highlights perceptually important regions.
    
    Parameters:
        img: Input image (float64, 0-1, BGR)
    
    Returns:
        WS: Saliency weight map
    """
    # Compute luminance
    L = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
    
    # Use a large Gaussian to approximate the global mean at each scale
    # Sigma proportional to image size for center-surround contrast
    sigma = max(L.shape) * 0.04
    kernel_size = int(sigma * 6) | 1  # Ensure odd kernel size
    kernel_size = max(kernel_size, 3)
    
    blurred_L = cv2.GaussianBlur(L, (kernel_size, kernel_size), sigmaX=sigma)
    
    WS = np.abs(L - blurred_L)
    
    return WS


def compute_saturation_weight(img):
    """
    Saturation weight (WSat) - Equation 14.
    
    WSat = sqrt(1/3 * [(R-L)² + (G-L)² + (B-L)²])
    
    Measures the deviation of each color channel from the luminance.
    Gives greater emphasis to highly saturated (colorful) regions,
    contributing to visually appealing fusion results.
    
    Parameters:
        img: Input image (float64, 0-1, BGR)
    
    Returns:
        WSat: Saturation weight map
    """
    # Compute luminance
    L = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
    
    # Extract channels
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    
    # Equation 14
    WSat = np.sqrt((1.0 / 3.0) * ((R - L)**2 + (G - L)**2 + (B - L)**2))
    
    return WSat


def compute_normalized_weight_maps(inputs, delta=0.1):
    """
    Computes and normalizes weight maps for all fusion inputs.
    
    For each input:
      Wk = WL + WS + WSat  (aggregate weight)
    
    Normalization (Equation 15):
      W̄k = (Wk + δ) / (Σ Wk + K·δ)
    
    Where δ = 0.1 is a regularization term ensuring each input
    contributes to the output.
    
    Parameters:
        inputs: List of input images for fusion
        delta: Regularization parameter (0.1)
    
    Returns:
        List of normalized weight maps
    """
    K = len(inputs)
    weight_maps = []
    
    for img in inputs:
        WL = compute_laplacian_contrast_weight(img)
        WS = compute_saliency_weight(img)
        WSat = compute_saturation_weight(img)
        
        # Aggregate weight map
        Wk = WL + WS + WSat
        weight_maps.append(Wk)
    
    # Normalize across all inputs (Equation 15)
    weight_sum = np.zeros_like(weight_maps[0])
    for w in weight_maps:
        weight_sum += w
    weight_sum += K * delta
    
    normalized = []
    for w in weight_maps:
        W_bar = (w + delta) / weight_sum
        normalized.append(W_bar)
    
    return normalized


# =============================================================================
# STEP 5: MULTISCALE LAPLACIAN PYRAMID FUSION (Equations 17-18)
# =============================================================================

def build_gaussian_pyramid(img, levels):
    """
    Builds a Gaussian pyramid by iteratively downsampling.
    
    At each level, a Gaussian filter is applied to reduce resolution
    by a factor of 2 in both spatial dimensions.
    
    Parameters:
        img: Input image (float64)
        levels: Number of pyramid levels
    
    Returns:
        List of images at each pyramid level
    """
    pyramid = [img.astype(np.float64)]
    current = img.astype(np.float64)
    
    for i in range(levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def build_laplacian_pyramid(img, levels):
    """
    Builds a Laplacian pyramid (Equation 17).
    
    Each level captures band-pass details at different spatial frequencies:
      L_l{I} = G_l{I} - upsample(G_{l+1}{I})
    
    The coarsest level stores the low-frequency residual.
    
    Parameters:
        img: Input image (float64)
        levels: Number of pyramid levels
    
    Returns:
        List of Laplacian pyramid levels
    """
    gaussian = build_gaussian_pyramid(img, levels)
    laplacian = []
    
    for i in range(levels):
        # Upsample the next (coarser) level to match current level dimensions
        target_size = (gaussian[i].shape[1], gaussian[i].shape[0])
        upsampled = cv2.pyrUp(gaussian[i + 1], dstsize=target_size)
        
        # Laplacian = current level - upsampled coarser level
        lap = gaussian[i] - upsampled
        laplacian.append(lap)
    
    # Append the coarsest Gaussian level as the residual
    laplacian.append(gaussian[-1])
    
    return laplacian


def collapse_laplacian_pyramid(pyramid):
    """
    Reconstructs an image from its Laplacian pyramid.
    
    Starting from the coarsest level, iteratively upsample and add
    band-pass details to reconstruct the full-resolution image.
    
    Parameters:
        pyramid: List of Laplacian pyramid levels
    
    Returns:
        Reconstructed image
    """
    result = pyramid[-1].copy()
    
    for i in range(len(pyramid) - 2, -1, -1):
        target_size = (pyramid[i].shape[1], pyramid[i].shape[0])
        upsampled = cv2.pyrUp(result, dstsize=target_size)
        result = upsampled + pyramid[i]
    
    return result


def multiscale_fusion(inputs, weight_maps, num_levels=None):
    """
    Performs multiscale fusion using Laplacian pyramids (Equation 18).
    
    P_l(x) = Σ_k G_l{W̄k(x)} · L_l{Ik(x)}
    
    Each source image is decomposed into a Laplacian pyramid,
    weight maps into Gaussian pyramids. Fusion occurs independently
    at each level, then the result is collapsed back into a full
    resolution image.
    
    This approach avoids halo artifacts from direct pixel-level fusion.
    
    Parameters:
        inputs: List of K input images (float64, 0-1)
        weight_maps: List of K normalized weight maps (float64)
        num_levels: Number of pyramid levels (auto-computed if None)
    
    Returns:
        Fused output image (float64, 0-1)
    """
    K = len(inputs)
    h, w = inputs[0].shape[:2]
    
    # Determine number of pyramid levels based on image dimensions
    if num_levels is None:
        num_levels = int(np.log2(min(h, w))) - 3
        num_levels = max(num_levels, 1)
        num_levels = min(num_levels, 8)
    
    # Build Laplacian pyramids for each input image
    input_lap_pyramids = []
    for img in inputs:
        lap = build_laplacian_pyramid(img, num_levels)
        input_lap_pyramids.append(lap)
    
    # Build Gaussian pyramids for each weight map
    weight_gauss_pyramids = []
    for w_map in weight_maps:
        gp = build_gaussian_pyramid(w_map, num_levels)
        weight_gauss_pyramids.append(gp)
    
    # Fuse at each pyramid level (Equation 18)
    fused_pyramid = []
    for level in range(num_levels + 1):
        fused_level = np.zeros_like(input_lap_pyramids[0][level])
        
        for k in range(K):
            weight = weight_gauss_pyramids[k][level]
            
            # Expand weight to 3 channels if needed
            if len(fused_level.shape) == 3 and len(weight.shape) == 2:
                weight = weight[:, :, np.newaxis]
            
            fused_level += weight * input_lap_pyramids[k][level]
        
        fused_pyramid.append(fused_level)
    
    # Reconstruct the fused image from the pyramid
    result = collapse_laplacian_pyramid(fused_pyramid)
    result = np.clip(result, 0.0, 1.0)
    
    # Post-fusion saturation enhancement to restore vibrancy
    # Convert to HSV, boost saturation, convert back
    result_uint8 = (result * 255).astype(np.uint8)
    hsv = cv2.cvtColor(result_uint8, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)  # Boost saturation by 40%
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # Slight brightness boost
    result_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    result = result_boosted.astype(np.float64) / 255.0
    
    return np.clip(result, 0.0, 1.0)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def underwater_image_enhancement(input_path, output_path):
    """
    Complete Underwater Image Enhancement Pipeline.
    
    Implements the 4-step process from the paper:
      1. Color Balancing (adaptive red/blue compensation + Gray World)
      2. Input 1: Morphological Processed Residuals (edge enhancement)
      3. Input 2: Normalized Unsharp Masking (sharpening)
      4. Multiscale Laplacian Pyramid Fusion (combining both inputs)
    
    Parameters:
        input_path: Path to the degraded underwater image
        output_path: Path to save the enhanced output image
    
    Returns:
        Enhanced image as uint8 numpy array (BGR)
    """
    print("=" * 60)
    print("  UNDERWATER IMAGE ENHANCEMENT (UIE)")
    print("  Paper: Scientific Reports (2026) 16:3182")
    print("=" * 60)
    
    # Read input image
    print(f"\n[1/5] Loading input image: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    print(f"      Image size: {img.shape[1]}x{img.shape[0]} pixels")
    
    start_time = time.time()
    
    # ====== STEP 1: Adaptive Color Balancing ======
    print("\n[2/5] Applying Adaptive Color Balancing...")
    print("      - Red channel compensation (Eq. 11)")
    print("      - Blue channel compensation (Eq. 12)")
    print("      - Gray World normalization")
    color_balanced = adaptive_color_balance(img)
    
    # Save intermediate result
    cb_path = os.path.join(os.path.dirname(output_path), "step1_color_balanced.png")
    cv2.imwrite(cb_path, (color_balanced * 255).astype(np.uint8))
    print(f"      Saved: {cb_path}")
    
    # ====== STEP 2: Input 1 — MPR ======
    print("\n[3/5] Computing Input 1: Morphological Processed Residuals...")
    print("      Parameters: t=0.01, s=3, c=1.0")
    input1_mpr = morphological_processed_residuals(
        color_balanced, t=0.01, s=3, c=1.0
    )
    
    # Save intermediate result
    mpr_path = os.path.join(os.path.dirname(output_path), "step2_mpr_input.png")
    cv2.imwrite(mpr_path, (input1_mpr * 255).astype(np.uint8))
    print(f"      Saved: {mpr_path}")
    
    # ====== STEP 3: Input 2 — Normalized Unsharp Masking ======
    print("\n[4/5] Computing Input 2: Normalized Unsharp Masking...")
    input2_sharp = normalized_unsharp_masking(color_balanced)
    
    # Save intermediate result
    sharp_path = os.path.join(os.path.dirname(output_path), "step3_sharp_input.png")
    cv2.imwrite(sharp_path, (input2_sharp * 255).astype(np.uint8))
    print(f"      Saved: {sharp_path}")
    
    # ====== STEP 4 & 5: Weight Maps + Multiscale Fusion ======
    print("\n[5/5] Performing Multiscale Fusion...")
    print("      - Computing Laplacian contrast weights")
    print("      - Computing saliency weights")
    print("      - Computing saturation weights (Eq. 14)")
    print("      - Normalizing weight maps (Eq. 15, delta=0.1)")
    
    # Compute normalized weight maps
    weight_maps = compute_normalized_weight_maps(
        [input1_mpr, input2_sharp], delta=0.1
    )
    
    # Save weight map visualizations
    for k, wm in enumerate(weight_maps):
        wm_vis = (wm / (np.max(wm) + 1e-10) * 255).astype(np.uint8)
        wm_path = os.path.join(os.path.dirname(output_path), f"weight_map_input{k+1}.png")
        cv2.imwrite(wm_path, wm_vis)
    
    # Perform multiscale Laplacian pyramid fusion
    print("      - Building Laplacian pyramids")
    print("      - Fusing at each pyramid level (Eq. 18)")
    result = multiscale_fusion([input1_mpr, input2_sharp], weight_maps)
    
    elapsed = time.time() - start_time
    
    # Save final result
    result_uint8 = (result * 255).astype(np.uint8)
    cv2.imwrite(output_path, result_uint8)
    
    print(f"\n{'=' * 60}")
    print(f"  Enhancement complete in {elapsed:.2f} seconds")
    print(f"  Output saved to: {output_path}")
    print(f"{'=' * 60}")
    
    return result_uint8


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_image = os.path.join(script_dir, "input.png")
    output_image = os.path.join(script_dir, "output.png")
    
    # Allow command-line arguments
    if len(sys.argv) >= 2:
        input_image = sys.argv[1]
    if len(sys.argv) >= 3:
        output_image = sys.argv[2]
    
    # Run enhancement
    enhanced = underwater_image_enhancement(input_image, output_image)
    
    print(f"\nIntermediate outputs also saved:")
    print(f"  - step1_color_balanced.png  (After color balancing)")
    print(f"  - step2_mpr_input.png       (Morphological Processed Residuals)")
    print(f"  - step3_sharp_input.png     (Normalized Unsharp Masking)")
    print(f"  - weight_map_input1.png     (Weight map for MPR input)")
    print(f"  - weight_map_input2.png     (Weight map for sharp input)")
