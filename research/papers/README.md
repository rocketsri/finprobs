# Paper series: log S-fBM theoretical investigation

Five self-contained LaTeX papers derived from the investigation in `../log-sfbm-unification-2026-07-12.md` and `../findings-summary-2026-07-12.md`. All share `finprobs.sty` for theorem/finding environments and the confidence-label convention (Established / Plausible but unverified / Speculative / Negative result) used throughout this project.

| File | Title |
|---|---|
| `paper0-synthesis.tex` | Theoretical Improvements to Unified Rough/Multifractal Volatility Models: A Synthesis |
| `paper1-distributional-vs-temporal.tex` | Distributional versus Temporal Multifractality in the log S-fBM Volatility Model |
| `paper2-aggregation-rg-flow.tex` | Cross-Sectional Aggregation and the Scale-Dependence of Roughness in log S-fBM |
| `paper3-microstructural-derivation.tex` | Toward a Microstructural Derivation of Cascade Parameters in Rough/Multifractal Volatility |
| `paper4-spx-vix-calibration.tex` | Does Unification Help? The Intermittency Parameter and Joint SPX/VIX Calibration |

PDFs are included pre-built. To recompile any paper (needs `texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-science`, which supplies `amsthm`, `natbib`, `booktabs`, `bm`):

```
pdflatex paper1-distributional-vs-temporal.tex
pdflatex paper1-distributional-vs-temporal.tex   # second pass resolves citations
```

Bibliographies are inlined via `thebibliography` (no external `.bib`/`bibtex` step needed) — two `pdflatex` passes are sufficient.
