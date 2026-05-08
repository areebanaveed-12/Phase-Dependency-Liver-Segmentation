# Dependency-Aware Liver Segmentation

This project studies **multi-phase liver tumor segmentation from CT** with a focus on **explicit phase interaction modeling**.

The work progressed across three implementation phases:

- **Phase 3:** single-phase U-Net baseline on LiTS
- **Phase 4:** four-phase Harvard baseline with an attention U-Net
- **Phase 5:** final dependency-aware multi-phase model with explicit phase interaction modeling

## Project Idea

Multi-phase abdominal CT contains four clinically meaningful phases:

- **NC**: non-contrast
- **AP**: arterial phase
- **PVP**: portal venous phase
- **DP**: delayed phase

Most multi-phase segmentation methods benefit from multiple phases, but often treat phase relationships **implicitly** through generic fusion.  
This project focuses on a stronger question:

**Can we explicitly study how different phases interact, which combinations are most useful, and whether all phases are always necessary?**

## Main Contribution

The final model performs:

- **phase-specific encoding**
- **learned phase gating**
- **explicit pairwise phase interaction modeling**
- **complete combinational dependency analysis on the fixed trained model**

One of the most important findings is that:

- the **all-phase setting was not the best tumor-performing configuration**
- some **reduced phase subsets outperformed the full 4-phase input**

This supports the main motivation of the project:

**multi-phase performance depends not only on having more phases, but on how those phases interact.**

## Final Findings

Final dependency-aware Harvard-100 test results:

- **Liver DSC:** `0.9435`
- **Tumor DSC:** `0.6975`

Learned average phase gate importance:

- **NC:** `0.6709`
- **AP:** `0.5903`
- **PVP:** `0.6134`
- **DP:** `0.6291`

### Complete Phase Combination Study

The final notebook evaluates **all 15 possible non-empty phase combinations** on the **fixed trained model**.

Important outcomes:

- **Best single phase:** `PVP` -> Tumor DSC `0.3353`
- **Best two-phase combination:** `PVP+DP` -> Tumor DSC `0.7209`
- **Best three-phase combination:** `AP+PVP+DP` -> Tumor DSC `0.7109`
- **All four phases:** `NC+AP+PVP+DP` -> Tumor DSC `0.6975`

This means:

- `PVP+DP` outperformed the all-phase setting
- `AP+PVP+DP` also outperformed the all-phase setting

So the project shows that **more phases are not automatically better**.

## Repository Structure

- `baselineModelImplemented.ipynb`
  - Phase 3 baseline U-Net on LiTS
- `proposedImprovement.ipynb`
  - Phase 4 Harvard multi-phase baseline
- `two_seed_attempt.ipynb`
  - final Phase 5 dependency-aware submission notebook
- `SOA_survey.pdf`
  - initial project survey and motivation
- `final_submission_report.pdf`
  - final report
- `DL Project Report Template/`
  - LaTeX source and report template files

## Final Model Summary

The final `DependencyAttentionUNet` uses:

- **2.5D input**
  - 3 neighboring slices per phase
  - 12 total input channels
- **shared per-phase encoder**
  - preserves phase-specific features before fusion
- **phase gates**
  - learns how much each phase should contribute
- **explicit interaction term**
  - models pairwise cross-phase feature relationships
- **attention-based decoding**
  - final liver/tumor segmentation output

## Why This Project Matters

This project is not only about reaching a good Dice score.  
It is also about understanding:

- which phases are most useful
- which combinations work best
- whether dependency-aware modeling gives more interpretable multi-phase behavior

That is why the combinational dependency analysis is a central part of the project.

## Authors

- **Haider Wajahat    27100252**   
- **Areeba Naveed     27100239**
