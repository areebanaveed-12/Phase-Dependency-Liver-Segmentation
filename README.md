# Dependency-Aware Multi-Phase Liver Tumor Segmentation with Explicit Phase Interaction Modeling

Dependency-Aware Multi-Phase Liver Tumor Segmentation is a deep-learning framework for liver and tumor segmentation from four-phase abdominal CT (NC, AP, PVP, DP). The framework preserves phase-specific representations, learns per-phase importance gates, and adds an **explicit pairwise cross-phase interaction term** before decoding. It substantially improves tumor segmentation in heterogeneous multi-phase CT settings, and it ships with a complete combinational dependency analysis on the fixed trained model across all 15 possible non-empty phase subsets.

![Qualitative segmentation results](figures/fig6_qualitative_predictions.png)

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Experimental Results](#experimental-results)
- [Citations](#citations)

## Project Overview

Multi-phase CT acquires the liver across four contrast phases — non-contrast (NC), arterial (AP), portal venous (PVP), and delayed (DP) — each revealing different enhancement behavior and lesion conspicuity. Most multi-phase segmentation methods benefit from the additional inputs but treat phase relationships **implicitly** through generic channel fusion, so they cannot directly answer which phases are most useful or how phases interact. This project addresses that gap through:

1. **Phase-specific encoding**: Each phase is encoded by a shared per-phase encoder, preserving phase identity until fusion.
2. **Learned phase gating**: Per-phase scalar gates weight every phase contribution before aggregation.
3. **Explicit pairwise interaction modeling**: Pairwise feature products are projected back into the fused bottleneck so the decoder reasons over phase agreement and complementarity.

A central finding is that the **all-phase configuration is not the best tumor-performing setting**: reduced subsets such as `PVP+DP` (`0.7209` tumor DSC) and `AP+PVP+DP` (`0.7109` tumor DSC) outperform the full four-phase input (`0.6975`). This demonstrates that multi-phase performance depends not only on having more phases, but on how those phases interact.

## Repository Structure

```
Phase-Dependency-Liver-Segmentation/
├── README.md                       # Project documentation
├── src/
│   ├── Phase-3.ipynb               # Phase 3: single-phase U-Net baseline on LiTS
│   ├── Phase-4.ipynb               # Phase 4: four-phase Harvard baseline (Attention U-Net)
│   └── Phase-5.ipynb               # Phase 5: final dependency-aware multi-phase model
├── figures/
│   ├── fig1_training_dynamics.png            # Training/validation loss & DSC curves
│   ├── fig2_test_dsc_summary.png             # Held-out test liver/tumor DSC
│   ├── fig3_phase_gate_importance.png        # Average learned phase-gate activations
│   ├── fig4_all_phase_combinations_tumor_dsc.png   # Tumor DSC across all 15 subsets
│   ├── fig5_best_phase_subset_summary.png    # Best subset per cardinality
│   └── fig6_qualitative_predictions.png      # Qualitative segmentation outputs
└── research-paper/
    ├── final-research-paper.pdf    # Final ICML-style report
    └── SOA_survey.pdf              # State-of-the-art survey that motivated the project
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- numpy
- nibabel
- matplotlib
- pandas
- scikit-image

### Running the Code

1. Clone this repository:

```bash
git clone https://github.com/areebanaveed-12/Phase-Dependency-Liver-Segmentation.git
cd Phase-Dependency-Liver-Segmentation
```

2. Install dependencies:

```bash
pip install torch torchvision numpy nibabel matplotlib pandas scikit-image
```

3. Open and run the final dependency-aware notebook:

```bash
jupyter notebook src/Phase-5.ipynb
```

The notebooks contain implementations of:

- Phase 3: Single-phase U-Net baseline on LiTS
- Phase 4: Four-phase Harvard multi-phase Attention U-Net baseline
- Phase 5: Final 2.5D Dependency-Aware Attention U-Net with all-15-subset analysis

## Experimental Results

The final 2.5D `DependencyAttentionUNet` was trained on the 100-patient Harvard multi-phase split with a focal CE + Dice + tumor focal Tversky loss (mixed `0.25 / 0.35 / 0.40`), AdamW, validation-tuned thresholds, horizontal-flip TTA, and liver-constrained post-processing.

### Final Held-Out Test Performance

| Metric                   | Phase 3 (LiTS, 1-channel) | Phase 4 (Harvard-50, 4-phase) | Phase 5 (Harvard-100, 2.5D Dependency) |
|--------------------------|---------------------------|-------------------------------|----------------------------------------|
| Liver DSC                | 0.9111                    | 0.8896                        | **0.9435**                             |
| Tumor DSC                | 0.3360                    | 0.5287                        | **0.6975**                             |

### Learned Phase-Gate Importance

Average gate activations on the Harvard-100 test set show that the model uses every phase, but not equally:

| Phase | Mean Gate |
|-------|-----------|
| NC    | 0.6709    |
| AP    | 0.5903    |
| PVP   | 0.6134    |
| DP    | 0.6291    |

![Phase-gate importance](figures/fig3_phase_gate_importance.png)

### Complete Combinational Dependency Analysis

The fixed trained model was evaluated under all 15 possible non-empty phase combinations on the held-out test split:

| Input Phases  | N | Liver DSC | Tumor DSC  |
|---------------|---|-----------|------------|
| NC            | 1 | 0.9183    | 0.1857     |
| AP            | 1 | 0.9320    | 0.3113     |
| PVP           | 1 | 0.9530    | 0.3353     |
| DP            | 1 | 0.9372    | 0.1935     |
| NC+AP         | 2 | 0.9356    | 0.5696     |
| NC+PVP        | 2 | 0.9367    | 0.7137     |
| NC+DP         | 2 | 0.9212    | 0.5655     |
| AP+PVP        | 2 | 0.9456    | 0.6863     |
| AP+DP         | 2 | 0.9399    | 0.6635     |
| PVP+DP        | 2 | 0.9322    | **0.7209** |
| NC+AP+PVP     | 3 | 0.9459    | 0.6887     |
| NC+AP+DP      | 3 | 0.9400    | 0.6394     |
| NC+PVP+DP     | 3 | 0.9344    | 0.6985     |
| AP+PVP+DP     | 3 | 0.9440    | 0.7109     |
| NC+AP+PVP+DP  | 4 | 0.9435    | 0.6975     |

![Best phase subset per cardinality](figures/fig5_best_phase_subset_summary.png)

Key observations:

1. **Best single phase**: `PVP` (`0.3353` tumor DSC) — consistent with portal venous contrast being central to lesion characterization.
2. **Best two-phase subset**: `PVP+DP` (`0.7209`) — outperforms the full four-phase input.
3. **Best three-phase subset**: `AP+PVP+DP` (`0.7109`) — also outperforms the full four-phase input.
4. **All four phases**: `NC+AP+PVP+DP` (`0.6975`) — strong, but not the absolute top tumor configuration.

The combination of shared per-phase encoding, learned phase gates, and explicit pairwise interaction modeling yields a robust framework for multi-phase liver tumor segmentation, achieving a **+0.17** absolute improvement in tumor DSC over the Harvard four-phase Attention U-Net baseline while exposing direct, interpretable evidence about which phase combinations are genuinely useful.

## Citations

If you use this work, please cite the following key references:

- Ronneberger, O., Fischer, P., and Brox, T. **U-Net: Convolutional Networks for Biomedical Image Segmentation.** *MICCAI*, 2015.
- Bilic, P. et al. **The Liver Tumor Segmentation Benchmark (LiTS).** *Medical Image Analysis*, 84:102680, 2023.
- Oktay, O. et al. **Attention U-Net: Learning Where to Look for the Pancreas.** *MIDL*, 2018.
- Xu, Y. et al. **PA-ResSeg: A Phase Attention Residual Network for Liver Tumor Segmentation from Multiphase CT Images.** *Medical Physics*, 48(7):3752–3766, 2021.
- Liu, Z. et al. **PA-Net: A Phase Attention Network Fusing Venous and Arterial Phase Features of CT Images for Liver Tumor Segmentation.** *Computer Methods and Programs in Biomedicine*, 244:107997, 2024.
- Wu, C. et al. **A Review of Deep Learning Approaches for Multimodal Image Segmentation of Liver Cancer.** *Journal of Applied Clinical Medical Physics*, 25(12):e14540, 2024.
- Lin, T.-Y. et al. **Focal Loss for Dense Object Detection.** *ICCV*, 2017.
- Abraham, N. and Khan, N. M. **A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation.** *ISBI*, 2019.

## Authors

- **Haider Wajahat** — 27100252
- **Areeba Naveed** — 27100239
