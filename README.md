# RadhaOmics

> Tissue-specific sex-bias detection and propagation scoring for omics datasets

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-green.svg)]()

---

## What is RadhaOmics?

RadhaOmics is the first tool to quantify not just *whether* your omics dataset is sex-biased, but *how much that bias distorts your biological conclusions* — tissue by tissue.

Most existing tools flag that bias exists. RadhaOmics computes the **BiasΔ score**: a tissue-aware propagation score that tells you exactly how much your differentially expressed genes would shift if your cohort were sex-balanced.

---

## The problem

Over 67% of published omics studies use male-skewed or sex-undisclosed datasets. Current normalisation methods (TMM, DESeq2) are blind to sex. The result: findings presented as universal that are actually male-specific.

This matters because:
- Drug dosing derived from male-biased data causes 2× higher adverse event rates in women
- The FDA (2023) and NIH SABV policy require sex-disaggregated reporting — but no tool enforces this at the analysis stage
- No existing tool accounts for tissue-specific sex-bias patterns

---

## The BiasΔ score (novel contribution)
BiasΔ = Σ(wᵢ × |FCbias − FCbalanced|) / n
Where:
- `wᵢ` = tissue-specific sex-bias weight from GTEx reference data
- `FCbias` = fold change in your imbalanced dataset
- `FCbalanced` = projected fold change in a sex-balanced cohort
- `n` = number of genes analysed

This score is tissue-aware: brain, liver, blood, and 34 other GTEx tissues each have their own sex-differential expression profile. A 78% male bias in liver data produces a completely different BiasΔ than the same imbalance in brain data.

---

## Features

- Sex ratio audit with M:F threshold checking
- Tissue-specific sex-bias profiling (37 GTEx tissues)
- BiasΔ propagation score — novel contribution
- Gene-level flags with fold change and p-values
- FDA SABV and NIH SABV compliance checklist
- Auto-generated methods section citation
- Support for RNA-seq, GWAS, proteomics, single-cell, microarray

---

## Status

🔬 Active development — independent research project

---

## Author

**Kirtana Prem** — Independent Researcher, MSc Bioinformatics & Data Science

Named for Anuradha — because women deserve to be in the data.

---

## Citation

If you use RadhaOmics in your research, please cite:

> Prem, K. (2025). RadhaOmics: Tissue-specific sex-bias propagation scoring for omics datasets. GitHub. https://github.com/KirtanaPrem/RadhaOmics
