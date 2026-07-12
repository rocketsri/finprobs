# Theoretical Improvements to Unified Rough/Multifractal Volatility Models

**A multi-agent research investigation into the log S-fBM model (Wu, Muzy & Bacry, 2022)**

Date: 2026-07-12
Method: ~26 independent Sonnet research/audit agents across 5 rounds (diverse exploration → registry/redirect → deepening + adversarial audit → cross-pollination → open-problem follow-up), plus this synthesis.

---

## Executive summary

The log S-fBM model unifies rough volatility (fBM-driven log-vol, Hurst *H*>0) and the Multifractal Random Walk / log-normal cascade (*H*→0) as two limits of one Gaussian process. This investigation does **not** find that the unification, by itself, is a theoretical advance beyond what Wu, Muzy & Bacry (2022) already showed. What it does find is a decisive **asymmetry in how well-supported the model's two dimensions are**:

- The **roughness (H) dimension** is comparatively solid: a real microstructural derivation exists at the pure-fBM endpoint (Hawkes → rough volatility, El Euch–Fukasawa–Rosenbaum), and a real, rigorously derived cross-sectional aggregation mechanism (Zarhali, Aubrun, Bacry, Bouchaud & Muzy, 2025/2026) explains *why* index-level volatility ends up smoother than any single name, given a smooth/rough factor split as an input.
- The **multifractal/intermittency (λ²) dimension is the weakest-supported part of the whole framework**, on three independent axes at once: it is not established whether the model's own multiscaling is genuinely temporal rather than a fat-tail artifact (Direction 1); no microstructural mechanism — Hawkes-based or otherwise — has been found that produces it (Direction 3); and a first-principles derivation shows λ² is *literally calibration-inert* for the one practical payoff (SPX/VIX joint calibration) most likely to justify it (Direction 4).

Two small, genuine derived results came out of this investigation that don't exist anywhere in the literature searched: (1) λ² cancels out of the ratio between short-maturity and VIX-maturity vol-of-vol at fixed *H*, so it cannot decouple SPX skew from VIX level — only the correlation-length parameter *T* can, and even that lever is undercut by the same aggregation mechanism that explains Direction 2; (2) two specific microstructural mechanisms (plain Hawkes superposition, quadratic/Zumbach Hawkes) were proven to be unable to produce genuine cascade structure, narrowing an open question to one precisely specified missing limit theorem. A repaired diagnostic (a boundary-corrected covariance likelihood-ratio test, nested via the model's own *H*→0 limit) is proposed as the most implementable way to actually test Direction 1 on real data — this has not been run.

No claim below is presented as more certain than the evidence supports. Every load-bearing citation was independently verified against the source (arXiv abstracts/full text fetched directly); one author misattribution and one maturity-range mischaracterization were caught and are corrected here.

---

## Ground truth: what the model actually is

A Round 3 agent retrieved and read the source paper directly (Wu, Muzy & Bacry, "From rough to multifractal volatility: the log S-fBM model," *Physica A* 2022, [arXiv:2201.09516](https://arxiv.org/abs/2201.09516)) rather than relying on the informal "Box-Cox/Tsallis deformation" description used in the original task framing. **That description is not accurate.** The actual construction:

- Log-volatility ω<sub>H,T</sub>(t) is a **stationary Gaussian process** (no separate nonlinear marginal deformation) with covariance
  C<sub>ω</sub>(τ) = (ν²/2)[T<sup>2H</sup> − τ<sup>2H</sup>] for |τ| < T, and 0 beyond T,
  where the three free, independently-specifiable parameters are **H ∈ (0, 1/2)** (roughness), **T > 0** (decorrelation/integral scale), and the **intermittency coefficient λ²**, linked to the process variance by ν² = λ²/[H(1−2H)].
- The volatility measure is M<sub>H,T</sub>(dt) = e<sup>ω<sub>H,T</sub>(t)</sup>dt.
- **fBM limit** (Prop. 1): as T→∞, ω<sub>H,T</sub> converges to standard fBM.
- **MRM/MRW limit** (Prop. 2): holding λ² and σ² fixed, as H→0 (forcing ν²→∞), M<sub>H,T</sub> converges to the classical Bacry–Muzy Multifractal Random Measure. The covariance itself converges to C<sub>ω</sub>(τ) → λ² log(T/τ) — an **exactly logarithmic** covariance, which is the actual mechanism generating multifractality (via Gaussian multiplicative chaos), not a Box-Cox-style curvature parameter.
- H and λ² are structurally **independent** free parameters (empirically mildly anti-correlated, but the paper explicitly does not claim a theoretical coupling).
- The paper is **purely a probability/statistics paper**: GMM estimators for H and λ² applied to realized-volatility data (24 indices, 296 stocks). It contains **no options-pricing, SPX, or VIX content** — the authors flag "a faithful model for asset and option prices" as explicit future work, never completed as far as this investigation found.
- The paper's own authors flag the index-vs-single-name aggregation question (this project's Direction 2) as an **open problem for future work**, not something they resolved — confirming Direction 2 is a genuine open research question, not a solved one.

This correction should replace the "Box-Cox/Tsallis deformation" framing in any future work referencing this model.

---

## Direction 1 — Distributional vs. temporal multifractality

**Question:** Is log S-fBM's multiscaling a genuine temporal cascade, or confounded by fat-tailed marginals?

**Label: Plausible but unverified for log S-fBM itself; Established as a regime-dependent phenomenon for the closely related rBergomi model.**

Brandi & Di Matteo (2026, [arXiv:2601.11305](https://arxiv.org/abs/2601.11305)) — independently verified against the paper's actual tables, not just its abstract — ran a two-stage surrogate test (fBM surrogates, then marginal-preserving shuffled surrogates) on rough Bergomi and found the distributional/temporal split is **sharply regime-dependent by H**: 95.3% distributional at H=0.001, 78.4% at H=0.01, crossing over to 70.9% temporal at H=0.05, and 90–96.5% temporal by H=0.1–0.2. They separately validated their method on classical MRW, finding it genuinely temporal-dominant (78.5% at λ=0.25) — that MRW data point is a **new empirical result**, not a re-confirmation of prior work; no earlier paper had run this test on MRW itself.

Two important caveats surfaced by follow-up work:
- **No one has run this test on log S-fBM itself** — every number above is for rBergomi or classical MRW, not the actual unified model across its H-range.
- A genuine tension exists with real-market (not model) studies: Zhou (2009) and Jiang, Xie, Zhou & Sornette (2012) find that *observed* market multifractality is mainly distributional/fat-tail-driven, and temporal correlations if anything *reduce* apparent multiscaling — the opposite emphasis from what Brandi & Di Matteo found for the MRW *model*. If log S-fBM is meant to model that empirical phenomenon, this mismatch between what the model produces and what markets actually show is an open question, not a resolved one. LeBaron (2001) adds a further identifiability caution: a simple additive (non-cascade) stochastic volatility model can mimic multifractal-looking statistics too, so multiscaling alone doesn't uniquely implicate cascade structure.

**A repaired diagnostic (usable, not yet run):** An initially-proposed rank/Gaussianization test was adversarially audited and found to have three real flaws (a Jensen-gap bias when using a smoothed volatility proxy; a strawman null that would misclassify genuine Gaussian-log-vol cascades as "no cascade"; an unspecified surrogate noise distribution). A follow-up agent repaired all three, arriving at a cleaner test: since the model's own H→0 limit shows the logarithmic covariance C<sub>L</sub>(τ)=λ²log(T/τ) is literally the boundary limit of the power-law family C<sub>P</sub>(τ;H) as H→0, the two are **nested via reparametrization**, not competing ad hoc models. This permits a proper boundary-corrected likelihood-ratio test (Self–Liang mixture-χ², since H=0 sits at the edge of the parameter space) of "H=0 (genuine cascade) vs. H>0 (self-affine)" directly on the estimated covariance function of a bias-corrected, devolatized log-vol proxy. This is easier to implement than the original structure-function approach (needs only covariance estimates at K log-spaced lags, not high-order moments) and needs roughly 10⁴–10⁵ observations spanning several decades of scale — a concrete, buildable next step that has not yet been executed.

**Practicality:** Directly checkable with real return/realized-volatility data once implemented; no proprietary data required.

---

## Direction 2 — Scale-dependence of H as an RG-like flow

**Question:** Is the empirical pattern H≈0.01 (single stocks) vs. H≈0.1 (indices) a derivable flow, or just fit?

**Label: Plausible but unverified — the aggregation *consequence* is rigorously derived; the aggregation *cause* is not.**

Zarhali, Aubrun, Bacry, Bouchaud & Muzy ([arXiv:2505.02678](https://arxiv.org/abs/2505.02678), since retitled "A Nested Factor Model for Equity Markets: Reconciling Multifractal Stock Returns and Rough Index Volatilities") take as an *empirical input* — imported from Wu-Muzy-Bacry (2022), not derived here — that a smooth common/market factor (H≈0.11) sits alongside "super-rough" idiosyncratic single-name log-vol (H≈0). Given that split, they **rigorously derive** why the index ends up smoother than every single constituent: idiosyncratic vol-of-vol contributions shrink under cross-sectional averaging (∝1/N<sub>s</sub>³ for equal-weighted portfolios) while the common factor's contribution does not, with an explicit, usable threshold condition **β⁴N<sub>s</sub> ≫ 10** determining when the common factor dominates. This is a real, checkable formula, not a fitted regularity — closing the specific gap an independent agent hit when it tried to derive the same result from scratch and got stuck needing to *assume* the smooth/rough split rather than derive it.

**What remains unresolved, found by an adversarial cross-check:** the single-name H≈0.01 input sits exactly inside the regime Direction 1 found to be dominated by distributional (not temporal) confound in the closely related rBergomi model. If that transfers to log S-fBM, the aggregation story's empirical anchor may itself be partly a statistical artifact rather than pure dynamics — this should be treated as an unresolved dependency, not a settled input. Two independent follow-up agents also tried to explain the smooth/rough split itself from first principles and both failed cleanly: cross-sectional averaging is proven to be a pure variance-reduction/selection mechanism (it reveals whatever the common factor already is; it cannot manufacture smoothness that isn't already there), and the temporal Hawkes-CLT mechanism that produces H>0 in the first place does not get smoother with more aggregated order flow — roughness is set by the kernel's tail index, not by sample size. The best surviving candidate explanation is an economic one (not yet found in the literature, but built from established pieces — the Mixture-of-Distributions Hypothesis, the Hawkes→rough-vol microstructural mapping, and Grigelionis-type point-process superposition theory): single-name order flow is driven by a small, tightly-coupled set of participants pushing it toward criticality, while market-wide flow superposes many weakly-correlated local feedback loops plus genuinely diffusive macro information, diluting effective endogeneity at the aggregate level. This yields concrete, falsifiable predictions (single-name Hawkes branching ratios should exceed the index's at matched resolution; the H-gap should track algo/HFT penetration) that no one has tested — but it directly conflicts with Filimonov & Sornette's finding that E-mini S&P 500 order flow is itself highly endogenous (>70%), so it should be labeled speculative, not adopted.

**A related cross-pollination result (Round 4):** the same diversification mechanism that raises index H likely also raises the index's effective decorrelation timescale T — but since Direction 4's math shows the SPX/VIX decoupling lever gets *weaker*, not stronger, as T grows, this means Direction 2's own mechanism would actively work against Direction 4's proposed fix at the index level, not reinforce it. This is a genuine, non-forced tension between two directions, with a new testable prediction attached (T<sub>index</sub> should open up specifically once β⁴N<sub>s</sub> ≳ 10, the same threshold Zarhali et al. derive for H).

**Practicality:** The β⁴N<sub>s</sub>≫10 threshold is a genuinely usable, falsifiable formula. No comparable formula exists yet for time-aggregation (as opposed to cross-sectional aggregation).

---

## Direction 3 — Microstructural derivation of H / λ²

**Question:** Can an order-flow mechanism derive the model's cascade parameters, not just fit them?

**Label: Established for H alone; Negative result for two specific candidate mechanisms toward λ²; open problem precisely narrowed but not solved.**

El Euch, Fukasawa & Rosenbaum (2018, *Finance and Stochastics*, [arXiv:1609.05177](https://arxiv.org/abs/1609.05177)) and Jaisson & Rosenbaum ([arXiv:1310.2033](https://arxiv.org/abs/1310.2033), [arXiv:1504.03100](https://arxiv.org/abs/1504.03100)) rigorously derive rough volatility (H = α−1/2, for α∈(1/2,1) the tail index of a nearly-unstable Hawkes kernel) as a scaling limit of order flow — a real, extended (through 2026) microstructural derivation, but only for the pure-fBM endpoint. **No paper derives the cascade/multifractal parameters (λ², T) from any microstructure** — this is a clean, established gap, not merely an unverified claim.

This investigation tested — and ruled out — three candidate mechanisms for closing that gap:

1. **Plain superposition of Hawkes processes at different timescales.** Proven (with an adversarial stress-test refining the exact conditions) to stay jointly Gaussian under time-invariant linear mixing, giving exactly linear ζ(q)=qH for log-volatility — capable of exotic long-range dependence, but not genuine multifractality. *Caveat found by the audit:* this negative result does **not** extend to σ<sub>t</sub>/returns themselves when log-vol is Gaussian with a log-correlated covariance (exactly log S-fBM's own H→0 structure) — that combination is the standard, already-known route to multifractality and doesn't need a hierarchy at all. It also doesn't hold if mixing weights are state-dependent.
2. **Quadratic Hawkes / Zumbach-effect models** ([arXiv:1907.06151](https://arxiv.org/abs/1907.06151), underlying Gatheral-Jusselin-Rosenbaum's quadratic rough Heston). These do escape Gaussianity, but into the wrong universality class — an affine/Pearson-diffusion (Heston-type) limit, not a log-normal cascade. A genuinely useful mechanism for leverage/Zumbach asymmetry, but a dead end for this specific question.
3. **Exactly-critical (not just near-critical) Hawkes processes**, checked against the rigorous classical theory (Cohen & Dombry 2024, [arXiv:2401.11495](https://arxiv.org/abs/2401.11495); Yaglom 1947). Exact criticality produces explosive ~t² variance growth, power-law long-range dependence, or linear Yaglom-type growth — never logarithmic covariance. An important conceptual clarification came out of this: "criticality" in the Hawkes branching-ratio sense and "criticality" in the GMC/dyadic-cascade sense are different mechanisms and should not be conflated.

A precisely specified open problem remains: whether pushing the *existing, established* single-Hawkes scaling limit to its own α→1/2 boundary (rather than needing a multiplicative hierarchy) could reproduce log S-fBM's log-covariance directly. No paper addresses this boundary at all — it's a genuine technical wall (the limiting kernel loses square-integrability exactly at H=0), not evidence either way. A closely adjacent, established result (Hager & Neuman, [arXiv:2008.01385](https://arxiv.org/abs/2008.01385)) shows that properly-renormalized fBM itself converges as H→0 to exactly the log-correlated field / GMC structure needed — via pure Gaussian-field renormalization, with no microstructure underneath. The specific missing object was pinned down precisely: a joint double-scaling limit theorem for a nearly-unstable Hawkes process with α<sub>n</sub>→1/2 *and* the subcriticality gap→0 at a coupled rate, such that the rescaled **log**-intensity (not the intensity itself) converges to a Gaussian process with degenerating covariance. This does not currently exist in the literature searched, in finance or in the broader point-process/mathematical-physics literature. A broader, non-Hawkes search (Kesten multiplicative recursions, branching-random-walk/GMC constructions, agent-based cascade models) found only pure mathematics with no market microfoundation, or phenomenological analogies fitted top-down to data — confirming no microstructural derivation of the cascade parameters exists anywhere yet.

**Practicality:** The established H-only result has a real (if noisy) empirical estimation route from tick data (branching ratio, kernel tail index) already used in practice (Hardiman-Bercot-Bouchaud, Jaisson-Rosenbaum calibrations). No estimation route exists for cascade parameters because no theory yet says what microstructure statistic would map to them.

---

## Direction 4 — Empirical/pricing payoff (SPX/VIX)

**Question:** Does unification actually help joint SPX/VIX calibration, or is it aesthetically tidier but practically inert?

**Label: Established that the joint-calibration problem is real and precisely characterized; Negative result (derived) that λ² specifically cannot be the fix; genuine gap that no one has tested log S-fBM on this at all.**

The joint SPX/VIX calibration failure of pure rough Heston/Bergomi is a well-established, precisely characterized problem (the "Guyon conjecture" territory): Gatheral, Jusselin & Rosenbaum ([arXiv:2001.01789](https://arxiv.org/abs/2001.01789)) pin the mechanism down exactly — steep short-dated SPX skew forces vol-of-vol so high it becomes inconsistent with observed VIX-implied-vol levels. Known fixes exist (quadratic rough Heston, Guyon-Lekeufack path-dependent volatility, quintic OU models, Gaussian polynomial Volterra models) — all add structure beyond plain fBM-driven log-vol. **No paper has tested log S-fBM, or any structurally similar unified model, against this problem** — confirmed independently by two separate literature searches finding nothing. A real cautionary precedent exists: Abi Jaber, Illand & Li ([arXiv:2212.08297](https://arxiv.org/abs/2212.08297)) found a conventional one-factor Markovian model *outperforms* rough and path-dependent alternatives with the same parameter count on this exact joint-calibration benchmark — extra structural sophistication does not automatically buy better fit.

This investigation went further than "untested" and actually derived whether log S-fBM's extra parameter *could* plausibly help, using the model's real covariance formula (not an assumed one). The result: at fixed H, λ² is a pure multiplicative amplitude on the covariance and **cancels out exactly** of the ratio between short-maturity (SPX) and ~1-month (VIX) vol-of-vol — so it structurally cannot be the lever that decouples short-dated skew steepness from VIX level, regardless of how the model is calibrated. This is a genuine derived result, not a guess, and it directly closes off the most natural hypothesis for why unification would help pricing. The one parameter that *does* change that ratio is T (the correlation/integral timescale) — structurally analogous to the second, slower timescale already used in existing multi-factor fixes elsewhere in the literature. But the Direction 2 cross-pollination above shows the same aggregation mechanism that makes the index smoother likely also pushes the index's effective T longer, which *weakens* rather than strengthens this lever at the index level — a real friction, not a synergy, between the model's two most promising theoretical threads.

**Practicality:** Fully feasible to test with public SPX/VIX option-chain data; the missing piece is a pricing/calibration engine for log S-fBM, which does not yet exist (the source paper is pure probability/statistics, with no options content).

---

## Cross-direction meta-finding

An adversarial audit run across all four directions together reached a decisive, non-obvious conclusion worth stating plainly: **the H/roughness dimension of log S-fBM is moderately supported by theory and data; the λ²/multifractal dimension is the weakest-supported component of the entire framework**, undermined simultaneously and independently on three fronts — theoretically (no microstructural derivation exists, and two specific candidate mechanisms were proven not to work), statistically (whether the model's own multiscaling is even genuinely temporal is unresolved, with a live tension against real-market evidence), and practically (λ² is derivably inert for the one calibration payoff most likely to justify it). Two independent derivation attempts (the aggregation-cause question in Direction 2, and the multiplicative-hierarchy question in Direction 3) hit structurally the same wall — both had to posit exogenous structure to get the desired result rather than deriving it — suggesting this may be a genuinely hard, not merely under-explored, class of problem.

The audit also flagged that the Gatheral-Jaisson-Rosenbaum "volatility is rough" (H≈0.1) result is cited in this project as an anchor for both Direction 2 (aggregation) and Direction 4 (calibration context) — these should be read as one empirical data point reused in two contexts, not as two independent corroborations.

---

## What would most efficiently move this forward

1. **Run the repaired covariance-functional-form diagnostic (Direction 1)** on real single-stock and index realized-volatility data — the most implementable, well-specified next empirical step in this whole investigation.
2. **Attempt the specific double-scaling Hawkes limit theorem (Direction 3)** — the missing object is now precisely stated, even though no one has proven it.
3. **Build a minimal log S-fBM option-pricing/calibration engine and test the T-lever hypothesis (Direction 4)** directly against public SPX/VIX data, with an explicit overfitting control (parameter-matched comparison, per the Abi Jaber-Illand-Li precedent) — this is the most direct way to find out whether unification has any practical payoff at all.
4. **Test the branching-ratio prediction (Direction 2)**: compare estimated Hawkes near-criticality for single names vs. the index at matched time resolution, to check the speculative economic mechanism proposed for why the common factor is smoother.

---

## Bibliography

- Wu, Muzy & Bacry (2022). "From rough to multifractal volatility: the log S-fBM model." *Physica A*. [arXiv:2201.09516](https://arxiv.org/abs/2201.09516)
- Zarhali, Bacry & Muzy (2026). "From rough to multifractal multidimensional volatility: A multidimensional Log S-fBM model." [arXiv:2601.10517](https://arxiv.org/abs/2601.10517) *(note: authors are Zarhali, Bacry & Muzy, not Wu — corrected here after an initial misattribution)*
- Zarhali, Aubrun, Bacry, Bouchaud & Muzy (2025/2026). "A Nested Factor Model for Equity Markets: Reconciling Multifractal Stock Returns and Rough Index Volatilities." [arXiv:2505.02678](https://arxiv.org/abs/2505.02678)
- Brandi & Di Matteo (2026). "Multiscaling in the Rough Bergomi Model: A Tale of Tails." [arXiv:2601.11305](https://arxiv.org/abs/2601.11305)
- Brandi & Di Matteo (2022). "Multiscaling and rough volatility: an empirical investigation." *Int. Rev. Financial Analysis* 84. [arXiv:2201.10466](https://arxiv.org/abs/2201.10466)
- Rak & Grech (2018). "Quantitative approach to multifractality induced by correlations and broad distribution of data." *Physica A*. [arXiv:1805.11909](https://arxiv.org/abs/1805.11909)
- Zhou (2009). "Finite-size effect and the components of multifractality in financial volatility." [arXiv:0912.4782](https://arxiv.org/abs/0912.4782)
- Jiang, Xie, Zhou & Sornette (2012). "Understanding the source of multifractality in financial markets." [arXiv:1201.1535](https://arxiv.org/abs/1201.1535)
- Jiang, Xie, Zhou & Sornette (2019). "Multifractal analysis of financial markets: a review." *Physics Reports*. [arXiv:1805.04750](https://arxiv.org/abs/1805.04750)
- LeBaron (2001). "Stochastic volatility as a simple generator of financial power-laws and long memory." *Quantitative Finance*.
- El Euch, Fukasawa & Rosenbaum (2018). "The microstructural foundations of leverage effect and rough volatility." *Finance and Stochastics* 22(2). [arXiv:1609.05177](https://arxiv.org/abs/1609.05177)
- Jaisson & Rosenbaum (2015). "Limit theorems for nearly unstable Hawkes processes." *Annals of Applied Probability* 25(2). [arXiv:1310.2033](https://arxiv.org/abs/1310.2033)
- Jaisson & Rosenbaum (2016). "Rough fractional diffusions as scaling limits of nearly unstable heavy tailed Hawkes processes." [arXiv:1504.03100](https://arxiv.org/abs/1504.03100)
- Gnabeyeu, Pagès & Rosenbaum (2026). "Fake stationary rough Heston volatility: microstructure-inspired foundations." [arXiv:2602.11032](https://arxiv.org/abs/2602.11032)
- El Karmi (2026). "Scaling Limits of Bivariate Nearly-Unstable Hawkes Processes and Applications to Rough Volatility." [arXiv:2605.03703](https://arxiv.org/abs/2605.03703)
- Hager, Horst, Wagenhofer & Xu (2026). "Microstructural Foundation of Rough Log-Normal Volatility Models." [arXiv:2603.13170](https://arxiv.org/abs/2603.13170)
- Blanc, Donier & Bouchaud (2017). "Quadratic Hawkes processes for financial prices." *Quantitative Finance*. [arXiv:1509.07710](https://arxiv.org/abs/1509.07710)
- Dandapani, Jusselin & Rosenbaum (2021). "From quadratic Hawkes processes to super-Heston rough volatility models with Zumbach effect." *Quantitative Finance*. [arXiv:1907.06151](https://arxiv.org/abs/1907.06151)
- Gatheral, Jusselin & Rosenbaum (2020). "The Quadratic Rough Heston Model and the Joint S&P 500/VIX Smile Calibration Problem." [arXiv:2001.01789](https://arxiv.org/abs/2001.01789)
- Cohen & Dombry (2024). "Functional Limit Theorems for Hawkes Processes." *Probability Theory and Related Fields*. [arXiv:2401.11495](https://arxiv.org/abs/2401.11495)
- Hager & Neuman (2020). "The Multiplicative Chaos of H=0 Fractional Brownian Fields." [arXiv:2008.01385](https://arxiv.org/abs/2008.01385)
- Bayer, Harang & Pigato (2021). "Log-modulated rough stochastic volatility models." *SIAM J. Financial Math.* [arXiv:2008.03204](https://arxiv.org/abs/2008.03204)
- Gatheral, Jaisson & Rosenbaum (2018). "Volatility is rough." *Quantitative Finance* 18(6). [arXiv:1410.3394](https://arxiv.org/abs/1410.3394)
- Bennedsen, Lunde & Pakkanen (2022). "Decoupling the short- and long-term behavior of stochastic volatility." *J. Financial Econometrics* 20(5). [arXiv:1610.00332](https://arxiv.org/abs/1610.00332)
- Bolko, Christensen, Pakkanen & Veliyev (2023). "A GMM approach to estimate the roughness of stochastic volatility." *J. Econometrics* 235(2). [arXiv:2010.04610](https://arxiv.org/abs/2010.04610)
- Fukasawa, Takabatake & Westphal (2019/2022). "Is volatility rough?" [arXiv:1905.04852](https://arxiv.org/abs/1905.04852)
- Granger (1980). "Long memory relationships and the aggregation of dynamic models." *J. Econometrics* 14(2).
- Zaffaroni (2004). "Aggregation and memory of models of changing volatility." *J. Econometrics*.
- Guyon (2022). "The VIX Future in Bergomi Models." *SIAM J. Financial Math.*
- Guyon & Lekeufack (2023). "Volatility Is (Mostly) Path-Dependent." SSRN:4174589.
- Gazzani & Guyon (2024/2025). "Pricing and Calibration in the 4-Factor Path-Dependent Volatility Model." *Quantitative Finance* 25(3). [arXiv:2406.02319](https://arxiv.org/abs/2406.02319)
- Abi Jaber & Li (2025). "Volatility Models in Practice: Rough, Path-Dependent, or Markovian?" *Math. Finance*. [arXiv:2401.03345](https://arxiv.org/abs/2401.03345) *(note: finds rough models underperform one-factor Markovian models specifically over 1-week-to-3-month maturities, and non-rough path-dependent/two-factor Markovian models outperform over the wider 1-week-to-3-year range — not a single blanket claim)*
- Abi Jaber, Illand & Li (2025). "Joint SPX-VIX calibration with Gaussian polynomial volatility models." *Math. Finance*. [arXiv:2212.08297](https://arxiv.org/abs/2212.08297)
- Rømer (2022). "Empirical Analysis of Rough and Classical Stochastic Volatility Models to the SPX and VIX Markets." *Quantitative Finance*.
- Filimonov & Sornette (2012, 2015). Reflexivity/endogeneity estimation via Hawkes branching ratios on E-mini S&P 500.
- Clark (1973); Tauchen & Pitts (1983); Epps & Epps (1976). Mixture of Distributions Hypothesis.

---

## Methodology note

~26 independent Sonnet agents were used across five rounds: (1) 8 diverse-exploration agents, two independent angles per direction, deliberately kept unaware of each other's approach; (2) manual registry-building and redirection by the orchestrator; (3) 10 deepening and dedicated adversarial-audit agents, checking citation accuracy, distributional/temporal conflation, circular derivations, and cross-direction consistency; (4) 4 cross-pollination agents connecting compatible threads across directions; (5) 8 follow-up agents on three problems the first four rounds newly opened up (a positive microstructural mechanism, the cause of common-factor smoothness, and the genuineness of log S-fBM's own multiscaling). Every load-bearing citation flagged as surprising or convenient was independently re-verified against the primary source rather than trusted from the proposing agent's summary; two accuracy issues were caught this way (an author misattribution, a mischaracterized maturity range) and are corrected in the bibliography above.
