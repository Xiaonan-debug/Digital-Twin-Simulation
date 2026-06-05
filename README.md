# Marginalize, Don't Vote
### Sampling‑Free Self‑Consistency for Diffusion LMs via Amortized Reasoning Marginalization

**A proposal written in the methodological style of SPG: take a quantity everyone estimates with a *naive, biased* procedure, characterize the bias precisely, replace it with a cheap *debiased estimator*, and show large gains on reasoning benchmarks.** SPG did this for the *policy gradient* (intractable likelihood → biased ELBO surrogate → sandwiched bound). This proposal does it for *test‑time reasoning aggregation* — and the fix exploits a property **only diffusion LMs have**.

> **The hook in one sentence.** A diffusion LM can answer a question *with its chain‑of‑thought still masked*, because masked‑diffusion training amortizes the marginalization over reasoning — so instead of sampling 64 chains and majority‑voting (the universal test‑time‑reasoning recipe), you can **read a debiased estimate of the reasoning‑marginalized answer posterior almost for free.** Autoregressive models structurally cannot do this.

---

## Table of Contents
1. [The quantity everyone is secretly estimating](#1-the-quantity-everyone-is-secretly-estimating)
2. [The DLM‑only fact that changes the game](#2-the-dlm-only-fact-that-changes-the-game)
3. [Where the cheap estimate is biased (the SPG move)](#3-where-the-cheap-estimate-is-biased-the-spg-move)
4. [The estimator: CV‑Marg](#4-the-estimator-cv-marg)
5. [Why this is not self‑consistency, Prophet, or temporal voting](#5-why-this-is-not-self-consistency-prophet-or-temporal-voting)
6. [Experiments & falsifiable claims](#6-experiments--falsifiable-claims)
7. [Risks](#7-risks)
8. [References](#references)

---

## 1. The quantity everyone is secretly estimating

For a question $q$ with latent reasoning $r$ and answer $a$, the *Bayes‑optimal* answer is the mode of the **reasoning‑marginalized posterior**

$$
p(a \mid q) \;=\; \sum_{r} p(a \mid q, r)\, p(r \mid q).
$$

**Self‑consistency (SC)** — the dominant test‑time reasoning method — is just a Monte‑Carlo estimate of this: sample $N$ chains $r_i \sim p(r\mid q)$, decode $a_i \sim p(a\mid q,r_i)$, and majority‑vote. SC works because more samples shrink the variance of an estimate of $p(a\mid q)$. Its cost is $N$ full generations (typically $N=$ 32–256). Every DLM test‑time‑scaling paper (dVoting, Temporal‑SC, HEX, Reward‑Guided Stitching) is a variance‑reduction or aggregation trick on top of *this same sampling loop*.

**Nobody has asked whether a diffusion LM even needs the sampling loop.**

## 2. The DLM‑only fact that changes the game

Recall the masked‑diffusion training objective (LLaDA, this folder, Eq. 3): the model is trained to predict masked tokens from a *randomly* masked context, with mask ratio $t\sim U[0,1]$. On reasoning data `[q] [r] [a]`, this means the model is *explicitly trained* on examples where **the entire reasoning $r$ is masked and it must still predict the answer $a$.** Formally, the masked predictor learns to approximate

$$
p_\theta\big(a \mid q,\, r{=}\textsf{[MASK]}\big)\;\approx\;\sum_{r} p(a\mid q,r)\,p(r\mid q)\;=\;p(a\mid q).
$$

That is the **reasoning‑marginalized posterior itself** — the very thing SC spends $N$ samples to approximate — available from **a single forward pass** by masking the rationale and reading the answer marginal.

This is **impossible for an autoregressive model.** An AR model conditioned on $q$ with no reasoning produces $p(a\mid q,\,r{=}\varnothing)$ = "answer without thinking," a *different and worse* distribution; it is not a marginalization, because AR never integrates over the unwritten future. The DLM's bidirectional, any‑order training is exactly what turns "reasoning masked" into "reasoning integrated out." **This is the unique structural asset the test‑time‑scaling literature has left on the table.**

## 3. Where the cheap estimate is biased (the SPG move)

If the amortized marginal were exact, we would be done — one forward pass replaces SC. It is not, and the value of the project is in **characterizing the bias precisely** (SPG's signature move) rather than hand‑waving it.

Two distinct, analyzable bias sources:

**(B1) The Jensen / mean‑field gap.** A single forward pass with $r$ masked performs an *internal, amortized* average. But the model is an imperfect integrator: passing a fully‑masked $r$ collapses the reasoning to a single "averaged context" and then predicts $a$, i.e. it computes something like $f(\mathbb{E}[r])$ rather than $\mathbb{E}[f(r)]$. By Jensen, for the nonlinear softmax readout $f$,

$$
\underbrace{p_\theta(a\mid q, r{=}\textsf{[MASK]})}_{\text{cheap, 1 pass}} \;\ne\; \underbrace{\mathbb{E}_{r\sim p_\theta(r\mid q)}\big[p_\theta(a\mid q,r)\big]}_{\text{true model marginal}} ,
$$

and the gap has a definite sign structure we can estimate (it systematically *under‑sharpens* on hard, multi‑modal‑reasoning instances where the averaged context is uninformative).

**(B2) Answer‑token factorization.** Decoding a multi‑token answer in parallel from the masked‑$r$ marginal assumes the answer tokens are conditionally independent — the same product‑of‑marginals error studied for parallel decoding. For short answer spans this is small and *measurable*.

The research contribution is to (i) prove these are the only first‑order error terms, (ii) give a cheap per‑instance estimate of their magnitude, and (iii) **debias** — turning a biased one‑pass estimate into a consistent one with a tunable compute dial, exactly as SPG turned a biased one‑sided bound into a low‑bias sandwiched one.

## 4. The estimator: CV‑Marg (control‑variate marginalization)

We interpolate between the cheap‑biased one‑pass marginal and expensive‑unbiased SC using the model's **own partial‑reasoning predictions as control variates**.

```text
CV-Marg (compute dial m ≪ N)
  1. p0  <- p_theta(a | q, r = [MASK])              # 1 pass: the amortized marginal (biased, free)
  2. for k = 1..m:                                   # m ≪ N cheap partial samples
        unmask a *small* random fraction ρ of r  (partial rationale r̃_k ~ q_ρ)
        p_k  <- p_theta(a | q, r̃_k)                  # a low-variance sample of the answer marginal
  3. # control-variate debias: the p_k bracket the Jensen gap; extrapolate ρ -> 1
     p_hat <- p0 + (1/m) Σ_k w(ρ) ( p_k - p0 )       # consistent as ρ,m grow; = p0 when gap is 0
  4. answer <- argmax p_hat ;  report calibrated confidence from spread of {p_k}
```

Intuition: each partial‑reasoning pass $p_k$ reveals how much the answer distribution *moves* as reasoning is filled in — that movement **is** the Jensen gap (B1). Averaging a handful of them, with a weight $w(\rho)$ derived from the noise schedule, extrapolates the masked‑$r$ estimate toward the true marginal. When the gap is zero (easy instances), $p_k\approx p_0$ and CV‑Marg costs ~1 pass; when it is large (hard instances), it spends a few more — an **adaptive, instance‑level compute allocation**, à la SPG spending estimation effort where the bound is loose.

Key properties to establish (the SPG‑style theory section):
- **Consistency:** $\hat p \to p(a\mid q)$ as $m,\rho\to$ full, recovering SC in the limit.
- **Bias bound:** $\lvert \hat p - p(a\mid q)\rvert$ bounded by a quantity estimable from $\{p_k\}$ (so we *know* per instance when one pass suffices).
- **Compute:** target accuracy at $m=4$–$8$ partial passes vs. $N=64$ full chains → an order‑of‑magnitude test‑time saving, *because each pass conditions on a partially‑revealed rather than fully‑sampled rationale.*

## 5. Why this is *not* self‑consistency, Prophet, or temporal voting

| Prior line | What it does | Why CV‑Marg is different |
|---|---|---|
| **Self‑consistency / dVoting / HEX** | Sample $N$ **full** chains, then vote | We **don't sample full chains**; we read the *amortized marginal* (reasoning masked) and debias it with $m\!\ll\!N$ *partial* passes. Different estimand mechanics, ~10× cheaper. |
| **Prophet** ("knows the answer early") | *When to stop* a single decode (early commit) | Orthogonal: about truncating *one* trajectory. We're about *aggregating the marginal* without trajectories. Composable with it. |
| **Temporal Self‑Consistency / "Time is a Feature"** | Aggregate the answer prediction **across denoising steps** of one decode | Those average *snapshots of a sampling trajectory*; we exploit the **masked‑training identity** $p_\theta(a\mid q,\textsf{[MASK]})\approx p(a\mid q)$ and a **principled bias correction**, not heuristic temporal averaging. |
| **SPG / DDPP** | Training‑time use of the intractable likelihood | We are **inference‑time**, **training‑free**, no reward model; same *spirit* (characterize a bias, debias cheaply) on a different problem. |

The contribution is a **theorem + estimator**: masked‑diffusion training secretly amortizes reasoning marginalization; the cheap readout is biased by an analyzable Jensen/factorization gap; CV‑Marg debiases it at a fraction of self‑consistency's cost. That is a new, DLM‑unique, training‑free test‑time‑reasoning method — not a tweak to the sampling‑and‑voting loop everyone shares.

## 6. Experiments & falsifiable claims

**Models.** LLaDA‑8B, Dream‑7B (off‑the‑shelf; training‑free method).
**Tasks.** GSM8K, MATH, ARC‑Challenge, CommonsenseQA, StrategyQA — tasks with an identifiable answer region (where "mask the rationale" is well‑defined).

- **C1 (the identity holds).** $p_\theta(a\mid q,\textsf{[MASK]})$ is a *meaningfully better* answer estimate than AR's no‑reasoning $p(a\mid q)$ at equal (one) forward pass — and is correlated with the SC marginal. *If the masked‑answer readout is no better than chance, the whole premise dies — this is the first, cheapest experiment to run.*
- **C2 (bias is real and structured).** The Jensen gap (B1) is measurable, has the predicted sign (under‑sharpening), and concentrates on high‑reasoning‑entropy instances.
- **C3 (CV‑Marg matches SC far cheaper).** CV‑Marg at $m=4$–$8$ partial passes matches or beats $N=64$ self‑consistency accuracy; the accuracy–compute curve dominates SC, dVoting, and Temporal‑SC at matched compute.
- **C4 (free calibration).** The spread of $\{p_k\}$ yields a calibrated confidence / abstention signal at no extra cost (selective‑accuracy gains over SC vote‑margin).
- **C5 (AR cannot do this).** The same recipe on an AR model (drop the rationale) *fails* — confirming the advantage is the diffusion marginalization, not prompt engineering.

**Ablations.** $\rho$ (reveal fraction) sweep; $m$ sweep; weight $w(\rho)$ from noise schedule vs. learned; answer‑span length vs. factorization error (B2).

## 7. Risks

- **The amortized marginal may be too biased to rescue cheaply** on some tasks → CV‑Marg gracefully degrades to SC as $m\to N$, so the worst case is "no worse than SC," and C1 tells us up front where one pass suffices.
- **"Identifiable answer region" requirement** limits v1 to tasks with a localizable answer (the same scope as Prophet); open‑ended generation is future work.
- **Factorization error (B2)** could dominate for long free‑form answers; mitigated by short‑answer focus and the dependency‑aware corrections now standard for parallel decoding.

---

## References

**From this folder**
- *Large Language Diffusion Models (LLaDA)* — the masked‑training objective that amortizes the marginalization (§2).
- *SPG: Sandwiched Policy Gradient* — the methodological template (characterize a bias from an intractable‑likelihood surrogate; debias cheaply). https://github.com/facebookresearch/SPG
- *Diffusion LMs Know the Answer Before Decoding (Prophet)* — orthogonal early‑exit; composable. https://github.com/pixeli99/Prophet

**Closest neighbors (positioned against in §5)**
- *Self‑Consistency Improves CoT Reasoning* — https://arxiv.org/abs/2203.11171
- *Diffusion of Thoughts (DoT)* — https://arxiv.org/abs/2402.07754
- *dVoting: Fast Voting for dLLMs* — https://arxiv.org/pdf/2602.12153
- *Time Is a Feature: Temporal Dynamics in Diffusion LMs* — https://huggingface.co/papers/2508.09138
- *Test‑Time Scaling via Reward‑Guided Stitching* — https://arxiv.org/abs/2602.22871
- *Test‑Time Scaling via Hidden Semi‑Autoregressive Experts (HEX)* — https://arxiv.org/html/2510.05040

---

*Same DNA as SPG — a precise bias in a quantity everyone estimates naively, a cheap principled estimator that fixes it, and reasoning‑benchmark validation — but on an inference‑time problem and exploiting a marginalization that is unique to diffusion language models. The make‑or‑break check (C1) is one forward pass on an existing checkpoint.*
