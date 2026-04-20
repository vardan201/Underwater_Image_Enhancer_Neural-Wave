# 🌊 Underwater Image Enhancement (UIE)

> A faithful Python implementation of the non-learning underwater image enhancement pipeline from the peer-reviewed paper published in **Scientific Reports (2026)**.

---

## 📄 Paper Reference

**"A robust and efficient non-learning approach for underwater image enhancement using color balancing and multiscale fusion"**

- **Journal:** Scientific Reports (2026), Vol. 16, Article 3182
- **DOI:** [https://doi.org/10.1038/s41598-025-33170-9](https://doi.org/10.1038/s41598-025-33170-9)

---

## 📁 Project Structure

```
RESEARCH/
│
├── underwater_enhancement.py      # Main enhancement pipeline (full implementation)
├── evaluate.py                    # Quantitative evaluation metrics (UCIQE, Entropy)
│
├── input.png                      # Raw degraded underwater input image
├── output.png                     # Final enhanced output image
├── Input_Output_Comparison.png    # Side-by-side visual comparison
│
├── Processed_images/              # Intermediate pipeline outputs
│   ├── step1_color_balanced.png   # After adaptive color balancing
│   ├── step2_mpr_input.png        # After Morphological Processed Residuals
│   ├── step3_sharp_input.png      # After Normalized Unsharp Masking
│   ├── weight_map_input1.png      # Fusion weight map for MPR input
│   └── weight_map_input2.png      # Fusion weight map for sharpened input
│
├── Research_Papers.pdf            # Supporting literature and references
└── U-Shape_Transformer_for_Underwater_Image_Enhancement (1).pdf  # Related paper
```

---

## 🧠 Algorithm Overview

The pipeline is entirely **non-learning** (no neural networks, no training data). It operates in **five sequential stages**, each grounded in a specific set of equations from the paper.

```
Input Image
    │
    ▼
[Step 1] Adaptive Color Balancing
    │   ├─ Red channel compensation   (Eq. 11)
    │   ├─ Blue channel compensation  (Eq. 12)
    │   └─ Gray World normalization
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[Step 2] Morphological           [Step 3] Normalized
         Processed Residuals              Unsharp Masking
         (Input 1, Eqs. 2–10)            (Input 2, Eq. 13)
    │                                  │
    └──────────────┬───────────────────┘
                   ▼
          [Step 4] Weight Maps
              ├─ Laplacian Contrast Weight
              ├─ Saliency Weight
              └─ Saturation Weight       (Eq. 14–15)
                   │
                   ▼
          [Step 5] Multiscale Laplacian
                   Pyramid Fusion        (Eqs. 17–18)
                   │
                   ▼
            Enhanced Output
```

---

## ⚙️ Pipeline — Step by Step

### Step 1 — Adaptive Color Balancing
Underwater images suffer from wavelength-dependent light attenuation — red light is absorbed most, making images appear blue-green. This step corrects that in two phases:

- **Red & Blue compensation** (Eq. 11–12): Each channel is boosted relative to the green channel, with spatial adaptivity — darker areas receive stronger correction.
- **Gray World normalization**: All channels are scaled to equal mean intensity, removing residual color casts. A soft 85/15 blend preserves natural color vibrancy.

### Step 2 — Morphological Processed Residuals (MPR)
Produces **Input 1** for fusion by enhancing edges while suppressing noise:

1. A Gaussian low-pass filter separates structure from detail (Eq. 2).
2. The residual (high-frequency detail) is extracted (Eq. 3).
3. Positive and negative components are separated (Eq. 4).
4. The **morphological operator M** (Eqs. 6–9) applies thresholding, area opening (removes tiny noise regions), and morphological reconstruction to preserve significant structural edges.
5. Contrast-controlled recombination produces the final MPR output (Eq. 10).

**Key parameters:** `t = 0.01` (amplitude threshold), `s = 3` (minimum region size), `c = 1.0` (contrast coefficient).

### Step 3 — Normalized Unsharp Masking
Produces **Input 2** for fusion with a parameter-free sharpening technique (Eq. 13):

```
S = (I + N{ I − G*I }) / 2
```

Where `G*I` is the Gaussian-blurred image and `N{·}` is global linear normalization across all channels (to preserve inter-channel color ratios). This mitigates scattering-induced blurring without the parameter sensitivity of classical unsharp masking.

### Step 4 — Weight Map Computation
Three complementary quality metrics are computed per-pixel for both inputs:

| Weight | Description |
|---|---|
| **Laplacian Contrast (WL)** | Absolute Laplacian of luminance — highlights edges and textures |
| **Saliency (WS)** | Center-surround luminance contrast — highlights perceptually important regions |
| **Saturation (WSat)** | RMS deviation of channels from luminance (Eq. 14) — favors colorful areas |

These are summed and normalized using a regularization term `δ = 0.1` (Eq. 15) to ensure each input always contributes to the fusion.

### Step 5 — Multiscale Laplacian Pyramid Fusion
Fuses the two enhanced inputs at multiple spatial frequencies (Eq. 17–18):

```
P_l(x) = Σ_k  G_l{W̄k(x)}  ·  L_l{Ik(x)}
```

- Each input image is decomposed into a **Laplacian pyramid** (band-pass at each level).
- Each weight map is decomposed into a **Gaussian pyramid**.
- Fusion happens independently at every pyramid level.
- The fused pyramid is collapsed back to produce the final image.
- A post-fusion saturation boost (+40%) restores color vibrancy after pyramid processing.

---

## 📊 Evaluation

`evaluate.py` quantitatively measures enhancement quality using two metrics:

### UCIQE — Underwater Color Image Quality Evaluation
A standard metric specifically designed for underwater imagery:

```
UCIQE = 0.4680 · σ_c  +  0.2745 · contrast  +  0.2576 · saturation
```

- **σ_c** — standard deviation of chroma (color richness variability)
- **contrast** — robust luminance range (1st–99th percentile)
- **saturation** — ratio of mean chroma to mean luminance

Higher UCIQE → better perceptual quality.

### Image Entropy
Measures the information content of the image using the Shannon entropy of the grayscale histogram. Higher entropy indicates richer detail and texture in the enhanced output.

### Running the Evaluation

```bash
python evaluate.py
```

Sample output:
```
=== INPUT IMAGE ===
Mean Red:   85.34
Mean Green: 110.22
Mean Blue:  128.45
Mean Saturation: 62.10/255
Contrast (StdDev): 38.70
Entropy: 6.821
UCIQE Score: 0.412

=== OUTPUT IMAGE ===
Mean Red:   124.56
Mean Green: 121.80
Mean Blue:  119.43
Mean Saturation: 95.32/255
Contrast (StdDev): 52.18
Entropy: 7.203
UCIQE Score: 0.531
```

---

## 🚀 Usage

### Basic Usage

```bash
python underwater_enhancement.py
```

Processes `input.png` and saves enhanced result to `output.png` by default.

### Custom Input/Output

```bash
python underwater_enhancement.py <input_path> <output_path>
```

**Example:**
```bash
python underwater_enhancement.py my_underwater_photo.jpg my_enhanced_photo.png
```

### Output Files Generated

After running, the following files are produced:

| File | Description |
|---|---|
| `output.png` | Final enhanced image |
| `step1_color_balanced.png` | Intermediate: after color balancing |
| `step2_mpr_input.png` | Intermediate: MPR-enhanced input |
| `step3_sharp_input.png` | Intermediate: sharpened input |
| `weight_map_input1.png` | Fusion weight map for MPR input |
| `weight_map_input2.png` | Fusion weight map for sharpened input |

---

## 🛠️ Requirements

```bash
pip install opencv-python numpy scipy scikit-image
```

| Library | Purpose |
|---|---|
| `opencv-python` | Image I/O, filtering, color space conversions, pyramid operations |
| `numpy` | Numerical array operations |
| `scipy` | Connected component labeling |
| `scikit-image` | Morphological reconstruction, area opening |

**Python:** 3.8 or higher recommended.

---

## 🔬 Key Design Decisions

- **No learning required** — The entire pipeline is classical signal processing; no GPU, no pretrained weights.
- **Global normalization in Unsharp Masking** — The detail signal is normalized globally across all three channels (not per-channel), which is critical for preserving color balance. Per-channel normalization would collapse all channels to the same range, effectively desaturating the image to grayscale.
- **Soft Gray World blending** — A hard Gray World correction can over-desaturate images. The implementation uses an 85/15 blend between the corrected and original images for a more natural result.
- **Regularized weight maps** — The `δ = 0.1` term in Eq. 15 ensures no input is completely excluded from fusion, even in uniform regions.
- **Post-fusion saturation boost** — Pyramid-based fusion slightly reduces perceptual saturation due to the averaging nature of the process. A 40% HSV saturation boost restores visual vibrancy.

---

## 📚 Related Work

| File | Description |
|---|---|
| `Research_Papers.pdf` | Curated supporting literature for this implementation |
| `U-Shape_Transformer_for_Underwater_Image_Enhancement.pdf` | Deep learning-based alternative approach for comparison |

---

## 📌 Notes

- Input images should be in standard 8-bit BGR/RGB format (`.png`, `.jpg`, `.jpeg`).
- The algorithm is designed for **real underwater imagery**; results on synthetic or non-underwater images may vary.
- Processing time scales with image resolution. Typical HD images process in under 5 seconds on a modern CPU.

---

