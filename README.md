# pypuf_TandV

pypuf_TandV extends [pypuf](https://github.com/nils-wisiol/pypuf) — the toolbox for
simulation, testing, and attacking Physically Unclonable Functions (PUFs) — with a
**physics-informed environmental modeling layer** based on the Alpha-Power Law
(APL) MOSFET delay model. It enables large-scale, pre-silicon evaluation of how
temperature and supply-voltage variation affect the reliability, uniqueness, and
machine-learning modeling resistance of delay-based PUF architectures — without
requiring transistor-level SPICE simulation for every operating point.

This is the core library: a cleaned-up implementation of the environmental delay
model and its integration into pypuf's evaluation pipeline. For worked examples,
scripts, and Jupyter notebooks reproducing the paper's experiments and figures,
see [pypuf-temp](https://github.com/harjas-kaur/pypuf-temp).

This repository accompanies the paper *"Physics-Informed Environmental Modeling of
PUF Architectures Using the PyPUF Framework,"* accepted as a short paper at
**IEEE ISVLSI 2026**.

## Motivation

PUFs derive device-unique cryptographic keys from intrinsic manufacturing
variation, but delay-based PUF responses are sensitive to temperature and
supply-voltage drift, which shifts internal delay margins and increases bit error
rates. Existing high-level simulators like pypuf model delay using an abstract
Linear Threshold Function (LTF) and do not capture physical temperature/voltage
dependence, while transistor-level (SPICE) simulation that does capture it is too
expensive to run at scale. pypuf_TandV closes this gap by injecting a physically
derived delay-scaling factor directly into pypuf's LTF evaluation pipeline,
enabling large-CRP-count environmental sweeps that would be impractical with
SPICE alone.

## Approach

The propagation delay dependence on temperature and supply voltage is modeled
using a relation derived from the Alpha-Power Law MOSFET model:

```
t_d ∝ (T^m / V_dd) · (V_dd − V_th)^(−α)
```

where `T` is absolute temperature, `V_dd` is supply voltage, `V_th` is threshold
voltage, `m` is the temperature mobility exponent, and `α` is the velocity
saturation index. From this, a dimensionless **delay-scaling factor** `S(T, V)` is
computed relative to nominal operating conditions and applied dynamically to the
LTF delay weights during each response computation — propagating environmental
effects through hierarchical PUF architectures while remaining compatible with
pypuf's existing metrics and ML-attack modules.

A `PhysicalFactors` module computes the scaling factor from temperature, supply
voltage, and technology parameters; this factor is applied inside a
`NoisyLTFArray`-compatible evaluation step, so any LTF-based PUF architecture
already supported by pypuf can be evaluated environmentally without changes to
the surrounding simulation pipeline.

## Reliability Metrics

Two complementary reliability definitions are implemented to separate systematic
environmental drift from intrinsic stochastic noise:

- **Local Reference Reliability (LRR):** compares responses to a reference
  measured at the *same* operating point — captures intrinsic noise only.
- **Global Reference Reliability (GRR):** compares responses to a reference
  measured once at nominal conditions (25°C, 1.2V) and re-evaluated across the
  full sweep — captures the combined effect of environmental drift and
  stochastic variability.

## Key Results

- **Setup:** `n = 64` challenge bits, 100,000 CRPs, evaluated across Arbiter,
  XOR Arbiter, Feed-Forward Arbiter, XOR Feed-Forward Arbiter, Permutation,
  Interpose, and Lightweight Secure PUF architectures. Full sweep range:
  0–140°C and 0.5–3V. Default parameters: `m = 1.5`, `α = 1.2` (NMOS) / `1.5`
  (PMOS).
- **Structural metrics** (bias/uniformity, uniqueness, similarity) stay close to
  their theoretical 50% ideal across the full environmental range (deviations
  under 0.15%), regardless of architecture.
- **Reliability** peaks near the nominal operating point and degrades as
  conditions move away from it. **Voltage variation causes larger reliability
  degradation than temperature variation** across every architecture tested.
- The **Feed-Forward Arbiter PUF** is the most environmentally robust
  architecture; the **Interpose PUF** is the most sensitive, since delay
  perturbations propagate through its sequential PUF stages.
- **ML modeling attacks** (Least Squares, LMN, Logistic Regression, MLP) were
  trained at nominal conditions (25°C, 1.2V) and evaluated across the full
  environmental sweep without retraining. The MLP attack was most effective
  (0.84 nominal accuracy) and was used for cross-environment analysis.
  Modeling accuracy is strongly dependent on operating conditions — lowest at
  high supply voltage and low temperature — but only weakly dependent on the
  delay-model parameters `m` and `α` themselves, indicating attack success is
  driven primarily by architectural structure rather than delay-model choice.

## Getting Started

This project builds on pypuf's simulation API. If you're new to pypuf itself,
start with the
[pypuf hello world](https://pypuf.readthedocs.io/en/latest/#getting-started) in
the [pypuf documentation](https://pypuf.readthedocs.org/).

```bash
git clone https://github.com/harjas-kaur/pypuf_TandV.git
cd pypuf_TandV
pip install -r requirements.txt
```

[[ Add a short, minimal usage example here, e.g. instantiating an
environment-aware Arbiter PUF via `PhysicalFactors` and evaluating it at a given
(T, V) point, once the public API is finalized. For fuller worked examples and
notebooks, point readers to pypuf-temp. ]]

## Repository Structure

[[ Fill in once finalized, e.g.:
- `pypuf_TandV/` — core library (delay-scaling model, `PhysicalFactors`,
  environment-aware LTF evaluation)
- `tests/` — unit tests
]]

## Paper

> H. Kaur, D. M. Das, and N. Goel, "Physics-Informed Environmental Modeling of
> PUF Architectures Using the PyPUF Framework," accepted at *IEEE International
> Symposium on VLSI (ISVLSI) 2026* (short paper).

[[ Add DOI / IEEE Xplore link once available. ]]

If this work is useful to you, please cite it as:

```bibtex
@inproceedings{kaur2026pypuftandv,
  author    = {Kaur, Harjas and Das, Devarshi Mrinal and Goel, Neeraj},
  title     = {Physics-Informed Environmental Modeling of {PUF} Architectures Using the {PyPUF} Framework},
  booktitle = {IEEE International Symposium on VLSI (ISVLSI)},
  year      = {2026},
  note      = {Short paper}
}
```

## Relationship to pypuf

pypuf_TandV is built on top of, and depends on, the original
[pypuf](https://github.com/nils-wisiol/pypuf) toolbox. If you use this
repository, please also cite pypuf itself:

> Nils Wisiol, Christoph Gräbnitz, Christopher Mühl, Benjamin Zengin, Tudor
> Soroceanu, Niklas Pirnay, Khalid T. Mursi, & Adomas Baliuka. pypuf:
> Cryptanalysis of Physically Unclonable Functions (Version 2, June 2021).
> Zenodo. https://doi.org/10.5281/zenodo.3901410

```bibtex
@software{pypuf,
  author       = {Nils Wisiol and
                  Christoph Gräbnitz and
                  Christopher Mühl and
                  Benjamin Zengin and
                  Tudor Soroceanu and
                  Niklas Pirnay and
                  Khalid T. Mursi and
                  Adomas Baliuka},
  title        = {{pypuf: Cryptanalysis of Physically Unclonable
                   Functions}},
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v2},
  doi          = {10.5281/zenodo.3901410},
  url          = {https://doi.org/10.5281/zenodo.3901410}
}
```

## Related pypuf Studies

pypuf has been used across a number of PUF-related research projects. A
selection, in reverse chronological order:

* 2021, Wisiol: [Towards Attack Resilient Arbiter PUF-Based Strong PUFs](https://eprint.iacr.org/2021/1004)
* 2021, Wisiol et al.: [Neural-Network-Based Modeling Attacks on XOR Arbiter PUFs Revisited](https://eprint.iacr.org/2021/555)
* 2020, Wisiol et al.: [Splitting the Interpose PUF: A Novel Modeling Attack Strategy](https://eprint.iacr.org/2019/1473)
* 2020, Wisiol et al.: [Short Paper: XOR Arbiter PUFs have Systematic Response Bias](https://eprint.iacr.org/2019/1091)
* 2019, Wisiol et al.: [Breaking the Lightweight Secure PUF](https://eprint.iacr.org/2019/799)
* 2019, Wisiol et al.: [Why Attackers Lose: Design and Security Analysis of Arbitrarily Large XOR Arbiter PUFs](https://doi.org/10.1007/s13389-019-00204-8)

See the full list, and the original pypuf toolbox, at
[nils-wisiol/pypuf](https://github.com/nils-wisiol/pypuf).

## Contact

Harjas Kaur — [harjaskaurbassi@gmail.com](mailto:harjaskaurbassi@gmail.com)
[GitHub](https://github.com/harjas-kaur) ·
[LinkedIn](https://www.linkedin.com/in/harjas-kaur-bassi/)

## License

[[ Add a license — pypuf itself is GPL-3.0; consider matching it if this
repository derives from pypuf's source. ]]
