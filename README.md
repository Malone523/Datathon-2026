# Datathon 2026 — Causal Approach to Scrap Reduction in Injection Molding

**Constructor University Bremen | Insytes Manufacturing Challenge 2026**  
Denis Rosu, Sonam Tobden, Tajwar Ali, Zamichael Mekasha

---

## Overview

This repository contains the code and paper for our submission to the 
Constructor University Datathon 2026. The challenge was to identify which 
variables in an industrial injection molding process drive scrap rate, and 
how they can be adjusted to improve it.

Rather than treating this as a purely predictive problem, we approached it 
through causal inference — using a predefined DAG and domain ontology to 
distinguish controllable process levers from contextual confounders and 
downstream outcomes.

The analysis identifies eight actionable process adjustments that jointly 
reduce the model-predicted scrap rate from a baseline of 4.44% to 
approximately 0.95%, representing a 78.6% relative improvement.

---

## Repository Contents

| File | Description |
|------|-------------|
| `prototype.py` | Interactive counterfactual simulator — adjust process variables and observe predicted scrap rate changes in real time |
| `formula_calculation.py` | Supporting calculations for causal effect estimation and the composite machine performance score |
| `final datathon paper.pdf` | Full research paper including methodology, DAG defence, results, and recommendations |

---

## Methodology

- **Causal graph (DAG):** Provided by Insytes domain experts, encoding the 
  five-layer causal structure of the injection molding process
- **Direct effects:** Multivariate OLS regression on direct causal parents 
  of scrap rate, yielding backdoor-adjusted estimates
- **Total effects:** Bivariate models for upstream indirect variables
- **Causal discovery check:** LiNGAM applied to a subsample as a 
  structural consistency check against the predefined DAG
- **Optimization:** Gradient Boosting Machine (R² = 0.58) combined with 
  Differential Evolution to identify optimal process settings

---

## Key Results

| Variable | Change | Scrap Δ |
|----------|--------|---------|
| Cooling time | 16.8 s → 24.6 s | −1.30% |
| Injection pressure | 1,182 bar → 900 bar | −0.92% |
| Tool wear index | 0.363 → 0.133 | −0.40% |
| Mold temperature | 69.4°C → 64.5°C | −0.34% |
| Resin moisture | 0.119% → 0.021% | −0.25% |
| Clamp force | 2,825 kN → 4,323 kN | −0.14% |
| Screw speed | 68.3 rpm → 88.1 rpm | −0.12% |
| Dryer dewpoint | −37.7°C → −44.5°C | −0.04% |

---

## Running the Prototype

```bash
pip install -r requirements.txt
python prototype.py
```

> Note: the dataset (`karcher_injection_molding_demo.csv`) is not included 
> in this repository as it was provided under the datathon's data usage terms.

---

## Acknowledgements

Claude Code (Anthropic, 2025) was used as an AI-assisted development tool 
during the implementation of the interactive prototype.
