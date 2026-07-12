# Findings Summary: log S-fBM Volatility Model Investigation

Companion summary to `log-sfbm-unification-2026-07-12.md` (full report) and `scripts/` (verification code). Organized by finding type rather than by the four original research directions, for quick scanning.

---

## Summary

The log S-fBM model (Wu, Muzy & Bacry, 2022) unifies rough volatility (H>0) and the multifractal cascade (H→0) as two limits of one Gaussian process. Across ~26 research/audit agents plus direct computational verification, the investigation found a clear asymmetry: the model's **roughness (H) dimension is comparatively well-supported** (a real microstructural derivation exists at the H-only endpoint; a real, now simulation-confirmed aggregation mechanism explains index-vs-single-name differences), while the **multifractal/intermittency (λ²) dimension is the weakest part of the framework**, undermined simultaneously on theoretical, statistical, and practical grounds. Two small but genuine new theoretical results were derived (not found in the literature searched), one new diagnostic was proposed and partially validated, and real data was pulled and analyzed directly rather than relying only on literature claims. Every load-bearing citation was independently verified against its source; two accuracy errors from source agents were caught and corrected before publication.

---

## Novel findings

Results and tools that did not exist anywhere in the literature searched before this investigation produced them.

1. **λ² is calibration-inert for SPX/VIX skew decoupling; T is the real lever.** Derived directly from the model's own covariance formula, then numerically confirmed exactly: the intermittency parameter λ² cancels out of the ratio between short-maturity (SPX) and ~1-month (VIX) vol-of-vol at fixed H, so it cannot be the extra degree of freedom that fixes the known joint-calibration problem. The correlation-length parameter T does control that ratio. No paper has applied log S-fBM to options pricing at all, so this is new.

2. **A repaired, implementable diagnostic for genuine-vs-artifact multiscaling.** Because the model's H→0 (cascade) covariance is literally the boundary limit of its H>0 (power-law) covariance family, the two are nested via reparametrization rather than competing ad hoc models — enabling a proper boundary-corrected likelihood-ratio test. This was proposed, then partially validated by Monte Carlo (see Negative findings below for its current limitation).

3. **A precisely specified missing theorem for the microstructural cascade derivation.** Rather than the vague "no one has derived λ² from microstructure," the investigation narrowed this to one specific object: a joint double-scaling limit theorem for nearly-unstable Hawkes processes (kernel tail index α→1/2 *and* the subcriticality gap→0, at a coupled rate) under which the **log**-intensity, not the intensity itself, would need to converge to a log-covariance Gaussian field. This does not exist in the finance or point-process literature searched, but it is now a concrete starting point rather than an open-ended gap.

4. **A new, falsifiable cross-direction prediction.** The same aggregation mechanism that makes index H smoother than single-name H likely also pushes the index's effective decorrelation timescale T longer — which, combined with finding #1, means this mechanism would work *against* the SPX/VIX T-lever, not for it. New testable prediction: T_index should open up at the same β⁴N_s≳10 threshold that governs H_index.

5. **A new, falsifiable economic hypothesis for why the common factor is smoother than idiosyncratic volatility.** Built from established pieces (Mixture-of-Distributions Hypothesis, the Hawkes→rough-vol microstructural mapping, Grigelionis-type point-process superposition theory) into a specific claim nobody has proposed: single-name order flow sits closer to Hawkes near-criticality than market-wide flow, which superposes many weakly-correlated feedback loops plus diffusive macro information. Comes with concrete, untested predictions (single-name branching ratios should exceed the index's at matched resolution). Flagged as genuinely uncertain — it conflicts with one existing empirical result (Filimonov & Sornette's high endogeneity estimate for E-mini S&P 500 flow).

6. **A ground-truth correction to the model's own definition.** The model is not the informal "Box-Cox/Tsallis deformation" commonly assumed (including in the original task framing) — it is the exponential of a stationary Gaussian process with covariance C(τ)=(ν²/2)[T^{2H}−τ^{2H}], with H, λ², T as three independently free parameters. This was verified by reading the source paper directly and should replace the informal framing going forward.

---

## Positive findings

Results that were confirmed, established, or successfully replicated — either by independent literature verification, derivation, or direct computation.

- **The H-only microstructural derivation is Established.** El Euch–Fukasawa–Rosenbaum and Jaisson–Rosenbaum rigorously derive rough volatility (H=α−1/2) as a scaling limit of a single nearly-unstable Hawkes process — a real, repeatedly-extended (through 2026) result, verified against primary sources.
- **The cross-sectional aggregation mechanism is derived and now simulation-confirmed.** Zarhali, Aubrun, Bacry, Bouchaud & Muzy (2025/2026) rigorously derive why index volatility ends up smoother than every single name, given a smooth/rough factor split, with an explicit usable threshold (β⁴N_s≫10). This was independently re-simulated from scratch in this investigation: aggregate H rose smoothly from ≈0.041 toward the common factor's H≈0.149 as N grew, tracking the paper's own threshold closely.
- **λ² cancellation / T-lever claim verified exactly by direct computation**, not just derived on paper (ratio invariant to λ² to 6 decimal places; strictly monotone in T; converges to 1 as T→∞ exactly as predicted).
- **"Index smoother than single names" replicated on fresh, independent real data.** 15 years of daily OHLC for the S&P 500 and 25 large-cap stocks (a dataset none of the cited papers used) reproduced the pattern cleanly with a standard structure-function-slope Hurst estimator: index Ĥ=0.094 vs. single-stock range 0.047–0.076.
- **The joint SPX/VIX calibration problem is Established and precisely characterized** in the literature (Gatheral, Jusselin & Rosenbaum's mechanism: short-dated SPX skew forces vol-of-vol inconsistent with observed VIX levels), independently confirmed by two separate literature searches.
- **Every citation flagged as surprising or load-bearing was independently re-verified against its primary source** (arXiv abstracts/full text fetched directly, not trusted from summaries) — including a 2026 paper with unusually specific quoted percentages, which checked out exactly against the source tables.
- **The repaired diagnostic correctly detects genuine cascades 78% of the time** in Monte Carlo testing against known ground truth (see Negative findings for its asymmetric weakness).

---

## Negative findings

Claims, mechanisms, and hypotheses that were specifically tested and ruled out or found lacking — reported plainly rather than omitted, per the original task's own standard.

- **Plain superposition of Hawkes processes cannot produce genuine multifractality.** Proven (and adversarially stress-tested) to remain jointly Gaussian under time-invariant linear mixing, giving only linear ζ(q)=qH — capable of exotic long-range dependence, not genuine cascade structure. (Caveat: does not extend to state-dependent mixing weights, and does not rule out multifractality via Gaussian log-vol with a log-correlated covariance, which is the model's own actual mechanism.)
- **Quadratic Hawkes / Zumbach-effect models are a dead end for this specific question.** They escape Gaussianity but land in the affine/Pearson-diffusion (Heston-type) universality class, not the log-normal cascade class log S-fBM needs. Useful for leverage/asymmetry effects, irrelevant here.
- **Exact criticality (not just near-criticality) in Hawkes processes does not produce logarithmic covariance.** Checked against rigorous classical theory (Cohen & Dombry 2024; Yaglom 1947): produces explosive ~t² growth, power-law long-range dependence, or linear Yaglom-type growth — never the log-covariance signature needed.
- **No non-Hawkes microstructural mechanism was found either.** Kesten multiplicative recursions, branching-random-walk/Gaussian-multiplicative-chaos constructions, and agent-based cascade models were all checked; each is either pure mathematics with no market microfoundation, or phenomenological analogy fitted top-down to data.
- **A pure "CLT-of-CLTs" cannot explain why the common factor is smoother than idiosyncratic volatility.** Proven that cross-sectional averaging is a variance-reduction/selection mechanism only — it reveals whatever the common factor already is, it cannot manufacture smoothness that isn't already there.
- **The repaired diagnostic has a real, asymmetric power limitation.** Monte Carlo validation against known ground truth found only 50% (chance-level) correct classification of a true H=0.10 power-law process, versus 78% for a true log-covariance cascade, at practical sample sizes (N=1,500). The underlying logic is sound but this implementation is not yet reliable enough to trust on real data without improvement.
- **A more careful statistical test complicates the "index smoother than single-name" finding.** Applying a nested boundary model-selection test (rather than the simple structure-function slope) to the same real data found most series — including the index — statistically indistinguishable from the H=0 boundary once penalized for the extra parameter. This is a genuine estimator-fragility finding, not a contradiction, but it means the underlying empirical claim is less robust than the simple estimator alone suggests.
- **No test of log S-fBM against real SPX/VIX data exists, and a cautionary precedent argues against assuming it would help.** Abi Jaber, Illand & Li found a conventional one-factor Markovian model outperforms rough and path-dependent alternatives at matched parameter count on this exact benchmark — extra structure does not automatically buy better fit.
- **One economic-mechanism candidate for common-factor smoothness (finding #5 above) is directly contradicted by existing evidence.** Filimonov & Sornette found E-mini S&P 500 order flow is itself highly endogenous (>70%), undercutting the naive version of the hypothesis.

---

## Future directions

Ranked by how directly they would move the open questions forward, given what is now known.

1. **Run the repaired diagnostic properly.** Use the full boundary-corrected (Self–Liang mixture-χ²) likelihood-ratio test rather than the simplified AIC comparison used here, on a larger sample (10⁴–10⁵ observations spanning several decades of scale) with a true high-frequency realized-variance proxy rather than the daily Parkinson-range estimator used in this pass. This is the most concrete, immediately actionable next step.
2. **Attempt the specific double-scaling Hawkes limit theorem** (α_n→1/2 jointly with the subcriticality gap→0, at a coupled rate) — the missing object for a positive microstructural derivation of the cascade parameters is now precisely stated, even though unproven.
3. **Build a minimal log S-fBM option-pricing/calibration engine and test the T-lever hypothesis directly** against public SPX/VIX option-chain data, with an explicit parameter-matched overfitting control per the Abi Jaber-Illand-Li precedent. This is the most direct way to find out whether unification has any practical payoff at all.
4. **Test the branching-ratio prediction** for why the common factor is smoother than idiosyncratic volatility: compare estimated Hawkes near-criticality for single names vs. the index at matched time resolution, using tick-level data if it becomes available.
5. **Re-run the real-data H comparison with true realized variance** (high-frequency data) instead of the daily range-based proxy used here, to see whether the "index smoother" pattern survives the more careful nested test once measurement noise is reduced — directly following up on the estimator-fragility tension this investigation surfaced.
6. **Test whether log S-fBM's own multiscaling (not just rBergomi's or classical MRW's) is genuinely temporal**, using the repaired diagnostic once validated, across the model's actual H-range rather than only at its endpoints.

---

For full derivations, citations, and the four-direction breakdown, see `log-sfbm-unification-2026-07-12.md`. For runnable code behind every computational claim above, see `scripts/`.
