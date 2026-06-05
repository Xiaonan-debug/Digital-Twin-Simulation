# Co‑Diffusion: Self‑Certifying Diffusion Language Models

**A research proposal for making correctness a _structural invariant_ of generation, not a post‑hoc behavior.**

> **One‑line thesis.** The durable advantage of diffusion language models (DLMs) is **not** speed — it is that a bidirectional, iteratively‑refined canvas lets a model generate an answer *and a certificate that the answer is correct* **at the same time, on the same canvas, under a trained mutual‑consistency constraint**. This dissolves the ordering dilemma that makes autoregressive (AR) self‑verification post‑hoc and unfaithful, and turns "is this output trustworthy?" into a quantity the model computes *by construction*.

---

## Table of Contents
1. [What I read, and the shape of the field](#1-what-i-read-and-the-shape-of-the-field)
2. [Motivation: the speed story is dying — find the durable prize](#2-motivation-the-speed-story-is-dying--find-the-durable-prize)
3. [The core gap nobody owns](#3-the-core-gap-nobody-owns)
4. [Problem formulation](#4-problem-formulation)
5. [Methodology](#5-methodology)
6. [Theoretical grounding](#6-theoretical-grounding-why-this-needs-a-diffusion-lm)
7. [Experimental plan](#7-experimental-plan)
8. [Why this is novel (explicit differentiation)](#8-why-this-is-novel-explicit-differentiation)
9. [Risks, limitations, and falsification](#9-risks-limitations-and-falsification)
10. [Broader impact](#10-broader-impact)
11. [References](#references)

---

## 1. What I read, and the shape of the field

I read the ten papers in this directory and surveyed ~30 additional recent works. The DLM literature has organized itself into five mature clusters:

| Cluster | Representative work (this folder + field) | What it optimizes |
|---|---|---|
| **Foundations / scaling** | LLaDA (8B masked diffusion rivals LLaMA‑3) | Can diffusion *be* an LLM at all |
| **Continuous / latent formulations** | LangFlow, ELF, Cola‑DLM, LaDiR | Embedding/latent‑space diffusion vs. discrete |
| **Inference efficiency** | Prophet (early‑commit), DPad (suffix dropout), d²Cache (KV cache), EDIT, Window‑Diffusion | Make DLMs *fast* |
| **Test‑time scaling / search** | S³, RFG, Reward‑Guided Stitching, Diffuse Thinking | Spend more compute for accuracy |
| **Alignment / steering / RL** | SPG (sandwiched policy gradient), DDPP (posterior steering), UniSteer | Make DLMs *aligned* |

Two papers in this folder are especially load‑bearing for my argument:

- **LLaDA** establishes the substrate: a non‑causal Transformer that predicts *all* masked tokens simultaneously and refines a full‑sequence canvas from `t=1` (all mask) to `t=0` (unmasked). Crucially, it breaks the **reversal curse**, the first hint that bidirectional refinement buys a *structural* capability, not just a different sampler.
- **"Diffusion LM Knows the Answer Before It Decodes"** (Prophet) shows **early answer convergence**: on GSM8K/MMLU the answer is internally fixed by ~half the steps. The field reads this as *"steps are wasted, cut them."* I read it as *"the canvas holds a stable belief about the answer long before it is emitted — so there is room, during the remaining steps, to make the rest of the canvas **prove** that belief."*

Everyone is mining the first four clusters for incremental speed/accuracy. The fifth (alignment) treats verification as something you *bolt on* with RL. **No one is using the canvas itself as a place where an answer and its justification co‑exist and constrain each other.** That is the gap I take.

---

## 2. Motivation: the speed story is dying — find the durable prize

The original pitch for DLMs was **parallel decoding ⇒ speed**. That pitch is now contested from inside the field:

- Critical evaluations ("How Efficient Are Diffusion Language Models, Really?") show the throughput advantage largely evaporates once quality is held fixed, because parallel decoding degrades quality and DLMs cannot natively KV‑cache. The efficiency cluster (d²Cache, DPad, Prophet) is essentially a multi‑year effort to *claw back* the speed that bidirectionality costs.
- The knowledge‑efficiency advantage ("Diffusion Beats AR in Data‑Constrained Settings") is real but is being commoditized into *dual‑objective* AR+MDM training, which an AR model can adopt without ever generating by diffusion.

**If speed and data‑efficiency are both being absorbed, what is the _durable_, AR‑inaccessible reason DLMs should exist?** I claim it is **structural verifiability**, and it follows directly from the one thing AR can never do: *let a later part of the output revise an earlier part.*

#### The faithfulness dilemma that AR cannot escape

Consider any task where an output should be *justified*: math (answer + reasoning), code (program + spec/tests), factual QA (claim + evidence), planning (plan + feasibility proof). An AR model must pick an order:

- **Justify‑then‑answer** (chain‑of‑thought first). The justification is committed *before* the answer exists, so it cannot be revised once the answer is known. Worse, the answer is then forced to be consistent with a rationale the model can no longer touch.
- **Answer‑then‑justify.** The justification is now *post‑hoc rationalization* — the well‑documented **CoT unfaithfulness** problem: the stated reasons are not the computational cause of the answer, and models will confidently fabricate support for a wrong answer.

The entire AR self‑verification literature (V1, CoVerRL, Chain‑of‑Verification, "Incentivize LLMs to self‑verify") lives inside this dilemma: it runs verification as a **second, sequential pass** and uses **RL to *incentivize*** honesty — a behavior we hope the model adopts, not a property of the architecture.

**A DLM has no order to pick.** On a bidirectional canvas, the answer region and the justification region are denoised *together*; bidirectional attention lets each condition on the other at every step. The answer can shape the proof *and the proof can shape the answer*, iterating to a mutually‑consistent fixed point. Faithfulness stops being a behavior to be incentivized and becomes a **constraint to be satisfied.** That is the durable, AR‑inaccessible prize — and no one is building for it.

---

## 3. The core gap nobody owns

> **Gap.** DLMs are the only language models that can hold, refine, and mutually constrain *two interdependent artifacts on one canvas*. The field has spent this capability entirely on making one artifact (the answer) faster or more accurate. It has never spent it on **co‑generating the answer together with a machine‑checkable certificate of the answer's own correctness, such that inconsistency between them is (a) trained toward zero and (b) read off at inference as a calibrated, *spatially‑localized* trust signal.**

This single mechanism, if it works, simultaneously attacks four open problems that are currently chased separately:

1. **Faithful reasoning / CoT** → the certificate *is* the reasoning, and it is structurally bound to the answer.
2. **Hallucination detection / UQ** (DynHD et al.) → residual claim↔certificate inconsistency is an *intrinsic* confidence signal, not an external probe.
3. **Self‑correction** (RemeDi, SCOPE) → inconsistency is the *signal that drives* which tokens to re‑mask, replacing hand‑tuned confidence heuristics with a principled "repair the contradiction" rule.
4. **Test‑time scaling** (S³, RFG) → extra compute is spent reducing a *meaningful* objective (the inconsistency field) rather than re‑converging to the same MAP estimate.

I am **not** proposing "DLM + a verifier model" (that would be A+B, and the AR world already does sequential generate‑then‑verify). I am proposing a *single model* that learns a generative process whose **fixed points are self‑consistent answer/certificate pairs.**

---

## 4. Problem formulation

### 4.1 Setup

Let a task instance be a prompt $c$ (e.g., a math problem, a function signature + docstring, a factual question). A DLM defines a denoiser $f_\theta$ that maps a partially‑masked sequence $x_t$ to per‑position distributions over the vocabulary $\mathcal V \cup \{\textsf{[MASK]}\}$.

We partition the generated canvas into two **strands** laid out on the *same* sequence and attended to *bidirectionally*:

- **Claim strand** $a$ — the answer/solution the user consumes (e.g., the final numeric answer + solution, or the program body).
- **Certificate strand** $z$ — a machine‑checkable witness that *entails or tests* the claim, drawn from a **task‑specific certificate grammar** $\mathcal G$ (examples below). $z$ is generated by the *same* model, not a separate one.

| Domain | Claim $a$ | Certificate $z$ (grammar $\mathcal G$) | Checker $V$ |
|---|---|---|---|
| Arithmetic / math | final answer + steps | a set of intermediate equalities / a verifier‑friendly derivation | symbolic equality / a tiny rule engine |
| Code | function body | unit tests + type/pre‑post conditions the body must satisfy | execute tests; type‑check |
| Constraint puzzles (Sudoku, Countdown) | filled solution | the per‑cell/per‑op constraints the solution claims to meet | constraint propagator |
| Factual QA | answer claim | cited supporting evidence spans + an entailment label | NLI / retrieval match |

The key property of $\mathcal G$: there exists a **checker** $V(a,z)\in[0,1]$ — *cheap, possibly non‑differentiable, possibly symbolic* — that returns how well claim $a$ satisfies its own certificate $z$. $V$ is used at **training time** (as a reward and to mine hard negatives) and, where available, at **inference time** as a guardrail; the model is trained to *internalize* it so $V$ is **not required** at inference.

### 4.2 Objective (informal)

We want a generative process whose samples $(a,z)$ are:

- **Plausible**: $p_\theta(a,z\mid c)$ is high (fluent, on‑task) — standard diffusion likelihood.
- **Self‑consistent**: $V(a,z)=1$ — the claim genuinely satisfies its own certificate.
- **Grounded**: $z$ is a valid certificate *of this prompt* (it cannot be a trivial tautology) — enforced by $\mathcal G$ and an entailment term $E(c,z)$.

Formally, we target the **product distribution**

$$
\pi(a,z\mid c)\;\propto\;p_\theta(a,z\mid c)\,\cdot\,\underbrace{\exp\big(\beta\,[\,V(a,z)+\lambda\,E(c,z)\,]\big)}_{\text{consistency \& grounding tilt}} ,
$$

i.e. a base DLM *steered* toward the self‑consistent, grounded slice of its own output space. This is deliberately the **same mathematical shape** as reward‑tilted MDM steering (cf. DDPP's Bayesian‑posterior view and SPG's sandwiched bounds), which means we can borrow their estimators — but here the "reward" is the model's *own internal consistency*, computed on a strand the model itself wrote. That self‑referential closure is what makes it new.

---

## 5. Methodology

### 5.1 Canvas layout and the dual‑strand forward process

A training example is the concatenation `[c] [SEP] [a] [SEP] [z]`. The forward (corruption) process differs from vanilla LLaDA in two deliberate ways:

1. **Asymmetric masking schedules per strand.** We draw *two* independent mask levels $t_a, t_z \sim U[0,1]$ for the claim and certificate strands. Training on the full $(t_a,t_z)$ square — including the corners — is what teaches the four inference modes we need:
   - $t_a\!\downarrow, t_z\!\uparrow$: *given the answer, write its certificate* (verification skill).
   - $t_a\!\uparrow, t_z\!\downarrow$: *given the certificate, write the answer* (constraint‑guided solving).
   - both high: *co‑generate from scratch* (the deployment mode).
   - both low: *check / repair* a nearly‑complete pair.

   AR cannot be trained on the off‑diagonal corners coherently; the bidirectional canvas can.

2. **Adversarial substitution corruption (ASC) — the crucial ingredient.** Standard masked diffusion *only ever maps `[MASK] → token`*; it never sees a *wrong, committed* token, so it never learns that an answer can contradict its certificate. We additionally corrupt a fraction of *unmasked* claim tokens by **substituting hard negatives**: plausible‑but‑wrong values mined from (i) the model's own earlier‑checkpoint samples that fail $V$, and (ii) perturbations that flip $V(a,z)$ from 1→0. The model is trained to *detect and repair* these — i.e., to drive the canvas back onto the $V\!=\!1$ manifold. This is the training signal that vanilla MDMs structurally lack, and it is what makes the inconsistency field at inference *mean something*.

### 5.2 Reverse process: co‑denoising to a consistency fixed point

At inference we start from a fully‑masked $(a,z)$ canvas and run co‑denoising. Each step the model emits, per position, both a token distribution **and** a scalar **inconsistency estimate** $u_i\in[0,1]$ from an auxiliary head (trained against the ASC labels and against $1-V$). The remasking rule is the heart of the method:

```text
Co-Diffusion decoding (one macro-step)
  1. Predict token distributions for all masked positions on BOTH strands (one bidirectional forward pass).
  2. Predict the inconsistency field u over ALL positions (committed + masked).
  3. Commit low-uncertainty AND low-inconsistency tokens (standard confidence ∧ consistency gate).
  4. RE-MASK already-committed tokens whose u is high  ← true revision, claim↔certificate driven.
  5. Halt when the global inconsistency  U = mean_i u_i  drops below τ  (learned fixed-point criterion),
     OR escalate compute (Step 6) if U plateaus above τ.
  6. (Test-time scaling) If stuck in a high-U local optimum: partial RE-NOISE of the highest-u contiguous
     region (push it back up the diffusion ladder) and re-settle — a "shake the contradiction loose" move.
```

Three things fall out for free:

- **Self‑correction without heuristics.** Step 4 re‑masks because of a *contradiction with the certificate*, not because of a hand‑set confidence threshold (contrast RemeDi/SCOPE, which must learn what "low quality" means with no anchor).
- **Calibrated, spatially‑localized trust.** The final inconsistency field $u$ tells the user *which spans to distrust*; the global $U$ gives a single calibrated confidence. Setting an abstention threshold on $U$ yields **selective generation** ("I'm not sure" instead of a confident wrong answer) — natively, with no extra UQ machinery (contrast DynHD, which trains a separate deviation detector).
- **Meaningful test‑time scaling.** Step 6 spends extra steps reducing $U$, a quantity correlated with correctness — so more compute buys *more proof*, giving the step‑count → accuracy scaling law the field has been trying to manufacture with external search.

### 5.3 Training objective

Total loss over the $(t_a,t_z)$ corruption square:

$$
\mathcal L(\theta)=\underbrace{\mathcal L_{\text{MDM}}^{a}+\mathcal L_{\text{MDM}}^{z}}_{\text{(1) reconstruct both strands}}
\;+\;\gamma\,\underbrace{\mathcal L_{\text{incons}}}_{\text{(2) predict } u \text{ vs. ASC/V labels}}
\;+\;\eta\,\underbrace{\mathcal L_{\text{repair}}}_{\text{(3) fix substituted tokens}}
\;+\;\rho\,\underbrace{\mathcal L_{\text{consistency-RL}}}_{\text{(4) drive } V(a,z)\to 1}.
$$

- **(1)** is the standard LLaDA cross‑entropy on masked tokens (Eq. 3 of LLaDA), applied to *both* strands — so the base model retains full DLM capability and the proposal is a *drop‑in finetune on any LLaDA/Dream‑class checkpoint*, not a from‑scratch effort.
- **(2)** trains the inconsistency head: a per‑position binary/regression target = "was this token an ASC hard‑negative?" ∪ "does the current pair fail $V$ here?".
- **(3)** is cross‑entropy *on substituted (non‑mask) positions* — the term that teaches token→token correction, absent from all standard MDMs.
- **(4)** is the consistency tilt. Because the DLM log‑likelihood is intractable, we estimate the policy gradient with **SPG‑style sandwiched bounds** (upper+lower bound the log‑likelihood of high‑$V$ vs low‑$V$ pairs), inheriting SPG's lower‑bias estimator wholesale. Reward $=V(a,z)+\lambda E(c,z)$ requires **no human labels** — it is computed by the symbolic checker on the model's own output, so this scales without annotation.

**Two‑phase recipe.** (a) *Cold start*: supervised co‑diffusion on datasets where certificates exist or are cheaply synthesizable (math with extractable equalities; code with generated tests; puzzles with their constraints). (b) *Self‑play refinement*: the model proposes $(a,z)$ pairs, the checker $V$ scores them, ASC hard‑negatives are refreshed from current failures, and phases (3)+(4) tighten the consistency fixed point — a label‑free virtuous cycle analogous to AR generator‑verifier co‑evolution, but realized *inside one bidirectional sampler* instead of two sequential passes.

### 5.4 Efficiency and compatibility

Co‑Diffusion is deliberately orthogonal to the efficiency cluster: the certificate strand roughly doubles canvas length, but (i) **DPad's** suffix‑dropout and (ii) **d²Cache's** adaptive KV reuse apply unchanged, and (iii) the certificate strand is *exactly* the kind of "scratchpad/reservoir" DPad already prunes aggressively, so the marginal cost is far below 2×. The learned halting on $U$ (§5.2) also tends to *shorten* runs on easy instances — recovering Prophet‑style early exit, but exiting on a *correctness* criterion rather than raw confidence.

---

## 6. Theoretical grounding: why this needs a diffusion LM

**Claim.** Faithful co‑generation of mutually‑constraining strands is *expressible* by a bidirectional masked‑diffusion model and *not* by a left‑to‑right AR factorization without an external second pass.

*Sketch.* A self‑consistent pair is a **fixed point** of the map $(a,z)\mapsto(\,\arg\!\max_a p(a\mid z,c),\ \arg\!\max_z p(z\mid a,c)\,)$. Reaching such a fixed point requires conditioning $a$ on $z$ **and** $z$ on $a$ within one generative object. An AR model factorizes $p(a,z\mid c)=\prod_i p(\cdot\mid \text{prefix})$ under a single linear order; whichever strand comes first is generated *without* the other and is then frozen, so the model can represent at most a *one‑directional* conditional, never the joint fixed point — exactly the faithfulness dilemma of §2, now stated as a representational fact. A masked DLM's reverse process, by contrast, is a sequence of bidirectional updates whose stationary points are precisely the joint‑consistent configurations; the consistency tilt of §4.2 reshapes the energy so those stationary points coincide with $V\!=\!1$. This also recovers LLaDA's reversal‑curse result as the trivial two‑token special case ("A is B" ⇔ "B is A" is the minimal mutual‑constraint), and *generalizes* it from token order to artifact‑level co‑determination.

The objective in §4.2 is a valid steering of the base DLM: it is the same reward‑tilted posterior that DDPP shows is approximable simulation‑free for MDMs, and the SPG sandwiched bounds give a low‑bias gradient — so the construction stands on existing, peer‑reviewed estimators rather than new unproven machinery.

---

## 7. Experimental plan

**Base models.** Finetune LLaDA‑8B (and Dream‑7B for replication) — no from‑scratch training needed.

**Tasks & checkers $V$.**
- **GSM8K / MATH** — certificate = intermediate equalities; $V$ = symbolic equality check.
- **HumanEval / MBPP** — certificate = generated unit tests + pre/post‑conditions; $V$ = execution + type‑check.
- **Sudoku / Countdown** — certificate = claimed constraints; $V$ = propagator. (SPG shows DLMs already over‑perform here, an ideal stress test.)
- **Factual QA (e.g., PopQA / long‑tail)** — certificate = evidence spans + NLI label; $V$ = retrieval/entailment.

**Headline hypotheses (each independently falsifiable).**
- **H1 (faithfulness).** Co‑Diffusion certificates are *causally* bound to answers: perturbing the certificate changes the answer far more than for AR answer‑then‑justify (measured via counterfactual interventions on the certificate strand). *If not, the structural‑faithfulness claim fails.*
- **H2 (intrinsic UQ).** Global inconsistency $U$ separates correct from incorrect outputs better (AUROC) than entropy/confidence baselines and than DynHD‑style external detectors, **with no extra trained detector.**
- **H3 (selective generation).** At matched coverage, abstaining on high‑$U$ yields strictly higher selective accuracy than AR self‑verify and than DLM confidence thresholding.
- **H4 (meaningful scaling).** Accuracy increases monotonically with re‑noise budget (§5.2 Step 6) and the gains correlate with $\Delta U$, whereas vanilla DLM extra steps saturate (the Prophet observation) — demonstrating compute that buys *proof*, not just refinement.
- **H5 (self‑correction without heuristics).** Inconsistency‑driven re‑masking recovers more injected ASC errors than confidence‑driven re‑masking (RemeDi/SCOPE‑style) at equal step budget.

**Ablations.** Remove ASC (→ show the inconsistency field becomes uninformative — the load‑bearing test); remove certificate strand (→ collapses to LLaDA, isolating the co‑generation effect); single‑strand sequential verify on the *same* DLM (→ isolates *bidirectional* vs. *sequential* verification); replace symbolic $V$ with a learned reward (→ test reliance on symbolic checkers).

**Primary baselines.** LLaDA + low‑confidence remasking; LLaDA + Prophet early‑commit; LLaDA + RemeDi self‑remasking; AR (LLaMA‑3‑8B) with Chain‑of‑Verification and with V1‑style joint self‑verification; SPG‑aligned LLaDA.

---

## 8. Why this is novel (explicit differentiation)

| Prior line | What it does | Why Co‑Diffusion is different (not A+B, not incremental) |
|---|---|---|
| **AR self‑verification** (V1, CoVerRL, Chain‑of‑Verification, self‑verify‑RL) | Generate, *then* verify in a second pass; RL *incentivizes* honesty | Verification is **simultaneous and bidirectional**; faithfulness is a **structural constraint**, not an RL‑induced behavior. Impossible in AR (§6). |
| **DLM self‑correction** (RemeDi, SCOPE, T2M remasking) | Re‑mask "low‑quality" committed tokens to revise one strand | Re‑masking is driven by an **explicit claim↔certificate contradiction**, not a heuristic confidence/quality score with no anchor. |
| **DLM UQ / hallucination** (DynHD, "Optimizing Decoding Paths by Uncertainty") | Train an external detector on denoising dynamics | The trust signal $U$ is the **same quantity being optimized during generation** — intrinsic, no separate detector, spatially localized. |
| **DLM test‑time scaling** (S³, RFG, Reward‑Guided Stitching, Diffuse Thinking) | External search/PRM over many trajectories | Extra compute reduces an **internal, self‑written consistency objective**; single‑sampler, no external PRM or trajectory bank. |
| **DLM RL/steering** (SPG, DDPP) | Align to *external* rewards | The "reward" is the model's **own internal consistency on a strand it wrote** — self‑referential, label‑free. (SPG/DDPP are *reused as estimators*, not the contribution.) |
| **DLM efficiency** (Prophet, DPad, d²Cache) | Make one strand faster | Orthogonal and **reused**; halting becomes a *correctness* criterion, not a confidence one. |

The contribution is a **new generative paradigm** — answer and certificate as one mutually‑constrained diffusion object — plus the **ASC training signal** and the **inconsistency‑field controller** that make it learnable and useful. It reframes four separate research threads as consequences of one mechanism that *only a bidirectional, iteratively‑refined DLM can implement.*

---

## 9. Risks, limitations, and falsification

- **Certificate‑gaming.** The model could co‑generate a trivially‑true certificate (tautology) that the answer satisfies vacuously. *Mitigations:* the grounding term $E(c,z)$ (certificate must be entailed by the prompt), a minimum‑informativeness penalty, and adversarial certificate audits. *Falsifier:* if $U$ stays low while accuracy drops, the certificate is being gamed → idea fails as stated.
- **Domains without cheap checkers.** Open‑ended writing has no symbolic $V$. Scope v1 to *verifiable* domains (math, code, puzzles, grounded QA); learned $V$ for soft domains is explicitly future work, not claimed now.
- **Cost.** ~2× canvas; argued down via DPad/d²Cache (§5.4), but must be measured, not assumed.
- **ASC distribution.** If hard negatives are unrealistic, the inconsistency head won't transfer to genuine model errors. The self‑play refresh (§5.3) is designed to keep negatives in‑distribution; H5/ablation directly tests this.

I would consider the program **falsified** if H1 (causal faithfulness) and H2 (intrinsic UQ) both fail — that would show the certificate strand is decorative rather than load‑bearing.

---

## 10. Broader impact

Verifiable‑by‑construction generation is most valuable exactly where LLMs are least trusted: clinical, legal, financial, and scientific settings, and autonomous agents that must *know when they don't know* and abstain. By making the trust signal **intrinsic and spatially localized**, Co‑Diffusion supports targeted human review ("check these three spans") instead of all‑or‑nothing trust. It also repositions DLM research away from a losing speed race toward a capability — structural verifiability — that autoregressive models cannot replicate, giving the paradigm a durable reason to exist.

---

## References

**Papers in this repository**
- *Large Language Diffusion Models (LLaDA)* — Nie, Zhu, You, et al., NeurIPS 2025.
- *Diffusion Language Models Know the Answer Before Decoding (Prophet)* — Li, Zhou, et al., ICLR 2026. https://github.com/pixeli99/Prophet
- *LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling* — Chen, Liang, Sui, et al. https://github.com/nealchen2003/LangFlow
- *ELF: Embedded Language Flows* — Hu, Qiu, et al. (MIT). https://github.com/lillian039/ELF
- *Cola: Continuous Latent Diffusion Language Model* — Guo, Zhao, et al. (ByteDance Seed). https://hongcanguo.github.io/Cola-DLM/
- *DPad: Efficient Diffusion Language Models with Suffix Dropout* — Chen, Huang, et al. (Duke), ICLR 2026. https://github.com/Crys-Chen/DPad
- *d²Cache: Accelerating Diffusion‑Based LLMs via Dual Adaptive Caching* — Jiang, Cai, Luo, et al. (SEU), ICLR 2026. https://github.com/Kamichanw/d2Cache
- *SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models* — Wang, Rashidinejad, et al. (Meta/MIT), ICLR 2026. https://github.com/facebookresearch/SPG
- *Steering Masked Discrete Diffusion via Discrete Denoising Posterior Prediction (DDPP)* — Rector‑Brooks, Hasan, et al., ICLR 2025.
- *UniSteer: Text‑Guided Flow Matching in Activation Space for Versatile LLM Steering* — Shi, Zhang, et al. (ShanghaiTech).

**Field context surfaced during the literature sweep**
- *Don't Settle Too Early: Self‑Reflective Remasking (RemeDi)* — https://arxiv.org/abs/2509.23653
- *Revise, Don't Freeze: Sampler‑Matched Training for Self‑Correcting MDLMs (SCOPE)* — https://arxiv.org/html/2606.01026
- *Targeted Remasking / Token‑to‑Mask Refinement* — https://arxiv.org/abs/2605.26436 , https://arxiv.org/abs/2604.18738
- *DynHD: Hallucination Detection for Diffusion LLMs via Denoising Dynamics* — https://arxiv.org/pdf/2603.16459
- *Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty* — https://arxiv.org/pdf/2512.21336
- *S³: Stratified Scaling Search for Test‑Time in DLMs* — https://arxiv.org/pdf/2604.06260
- *RFG: Test‑Time Scaling for Diffusion LLM Reasoning with Reward‑Free Guidance* — https://openreview.net/forum?id=Q5YGlns7Bb
- *Test‑Time Scaling with DLMs via Reward‑Guided Stitching* — https://arxiv.org/abs/2602.22871
- *Diffuse Thinking: DLMs as Efficient Thought Proposers* — https://arxiv.org/pdf/2510.27469
- *How Efficient Are Diffusion Language Models, Really?* — https://arxiv.org/html/2510.18480v1
- *Diffusion Beats Autoregressive in Data‑Constrained Settings* — https://arxiv.org/pdf/2507.15857
- *Closing the Data‑Efficiency Gap Between AR and Masked Diffusion LLMs* — https://openreview.net/forum?id=J2J2LuXJev
- *Diffusion‑Inspired Masked Fine‑Tuning for Knowledge Injection* — https://arxiv.org/abs/2510.09885
- *V1: Unifying Generation and Self‑Verification for Parallel Reasoners* — https://harmandotpy.github.io/v1-verification/
- *CoVerRL: Generator‑Verifier Co‑Evolution* — https://arxiv.org/pdf/2603.17775
- *Chain‑of‑Verification Reduces Hallucination* — https://arxiv.org/pdf/2309.11495
- *Awesome‑DLMs survey* — https://github.com/VILA-Lab/Awesome-DLMs

---

*Prepared after reading all ten DLM papers in this directory and surveying ~30 recent related works. The proposal is intentionally a drop‑in finetune on existing LLaDA/Dream checkpoints and reuses peer‑reviewed estimators (SPG, DDPP) so that the novel claim — structural, co‑generated verifiability — can be tested cheaply and falsified cleanly.*

