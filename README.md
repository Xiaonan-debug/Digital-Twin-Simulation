# Relaxation Diffusion: Diffusion Language Models as Neural Constraint‑Propagation Reasoners

**A research proposal that takes the most striking — and unexplained — result in the papers in this folder and turns it into a new reasoning paradigm.**

> **TL;DR.** In the SPG paper (this folder), a diffusion LM beats a state‑of‑the‑art autoregressive RL model by **+27.0% on Sudoku** and **+18.4% on Countdown** — while the same models are nearly tied on GSM8K (+3.6%). That is not a small alignment delta; it is a *qualitative* gap that appears **only on constraint‑satisfaction / search problems**. No paper explains it. My thesis: **the denoising process of a diffusion LM is implicitly performing parallel constraint propagation**, a reasoning mechanism autoregressive (AR) models structurally cannot run. I propose to (1) establish this mechanistically, (2) show current DLMs do it only *weakly and accidentally*, and (3) build **Relaxation Diffusion** — a training + inference paradigm that makes the propagation **explicit, iterative, and learnable** by *decoupling belief‑refinement rounds from token‑commitment events* — yielding a general "reason‑by‑relaxation" engine for any language‑expressed constraint problem, with the size‑generalization that current DLMs lack.

---

## Table of Contents
1. [The observation that started this](#1-the-observation-that-started-this)
2. [Why AR cannot do this, and DLMs accidentally can](#2-why-ar-cannot-do-this-and-dlms-accidentally-can)
3. [The gap: implicit, weak, unscalable propagation](#3-the-gap-implicit-weak-unscalable-propagation)
4. [Problem formulation](#4-problem-formulation)
5. [Method: Relaxation Diffusion](#5-method-relaxation-diffusion)
6. [Why this is novel (not A+B, not a decoding tweak)](#6-why-this-is-novel)
7. [Experimental plan & falsifiable hypotheses](#7-experimental-plan--falsifiable-hypotheses)
8. [Risks and limitations](#8-risks-and-limitations)
9. [Alternative seeds (if you want a different flavor)](#9-alternative-seeds)
10. [References](#references)

---

## 1. The observation that started this

Reading the ten papers here, almost every result is a *quantitative* improvement on a shared axis (speed, alignment, perplexity). One result is **qualitatively different** and is buried in a table:

| Benchmark (SPG, Fig. 1 / Table 1) | Best AR‑style DLM‑RL baseline | SPG | Gap |
|---|---|---|---|
| GSM8K (arithmetic) | 82.5 | 86.1 | **+3.6** |
| MATH500 | 37.4 | 40.0 | +2.6 |
| **Countdown** (reach a target number) | 52.3 | 70.7 | **+18.4** |
| **Sudoku** (constraint grid) | 67.0 | 94.0 | **+27.0** |

The same training algorithm yields a **single‑digit** gain on free‑form arithmetic and a **27‑point** gain on Sudoku. The differentiator between these tasks is not "math vs. not‑math." It is that **Countdown and Sudoku are global constraint‑satisfaction problems**: every cell/operation is coupled to many others, and a commitment made early must be consistent with constraints that only become visible later.

LLaDA (this folder) hands us the second clue: it **breaks the reversal curse**, i.e. it can satisfy a constraint ("B precedes A") that an AR model, having committed to an order, cannot. Prophet (this folder) hands us the third: the canvas holds a *globally stabilizing belief* that settles over many refinement steps. Put together, these point at one mechanism the field has not named:

> **Hypothesis (Propagation Hypothesis).** A masked diffusion LM solves a constraint problem by maintaining a *full‑canvas partial assignment* and iteratively revising it so that local predictions become globally consistent — i.e., its denoising trajectory is an *amortized, learned constraint‑propagation / relaxation algorithm*. The huge Sudoku/Countdown gaps are the signature of this mechanism; the small GSM8K gap is what you get when global coupling is weak and a sequential chain suffices.

If true, this reframes what DLMs are *for*. The field sells DLMs on **speed** (now contested) and **likelihood**. The Propagation Hypothesis says the durable prize is a **different reasoning substrate**: parallel relaxation instead of sequential chaining.

---

## 2. Why AR cannot do this, and DLMs accidentally can

Consider a constraint problem with variables $v_1,\dots,v_n$ and constraints $C$ coupling them.

- **Autoregressive.** AR factorizes $p(v_1)\,p(v_2\mid v_1)\cdots$ under one fixed order and **commits irreversibly**. When it sets $v_2$, the constraints involving $v_7$ are not yet representable as conditioning context, so $v_2$ is chosen blind to them. If $v_7$ later forces $v_2$ to change, AR cannot go back. Chain‑of‑thought helps only by *serializing* the search into the context window — it is a single, linear search trajectory with no parallel propagation. This is the same structural limitation as the reversal curse, generalized from 2 variables to $n$.
- **Masked diffusion.** A DLM holds *all* $n$ variables on one canvas and, at every step, re‑predicts each from the **bidirectional** context of all currently‑committed others. A commitment to $v_7$ immediately changes the prediction for $v_2$ on the next step. Low‑confidence remasking even lets the model *defer* coupled variables until enough constraints have propagated. This is, structurally, one sweep of **iterative refinement over a factor graph** — exactly the shape of belief propagation / WalkSAT‑style local search.

So DLMs get constraint propagation **for free, as a side effect of bidirectional iterative denoising.** That is why SPG sees 27 points on Sudoku. But — crucially — they get it *accidentally and weakly*, which is the opening for real research.

---

## 3. The gap: implicit, weak, unscalable propagation

Current DLMs conflate two things that constraint solvers keep separate:

| Constraint solver | Current DLM | Problem |
|---|---|---|
| **Propagation step** (refine beliefs / prune candidate values, *no commitment*) | — fused into the same forward pass that also commits tokens | The model gets **one** propagation round per token revealed, then must commit. Hard problems need *many* propagation rounds before *any* safe commitment. |
| **Decision/commitment** (assign a variable) | low‑confidence remasking commits the *most certain* token each step | Commits the *easy* variables first (least information), starving the propagation that would resolve the *hard*, coupled ones. |
| **Backtracking** | once unmasked, frozen | A wrong early commitment is unrecoverable; error compounds — the well‑known accumulation problem. |

The result: implicit propagation that works on small grids (Sudoku 4×4/9×9) but **does not scale** with constraint density or problem size, and that no one has tried to strengthen *as propagation*. The self‑correction line (RemeDi, SCOPE) patches the backtracking row only, with confidence heuristics and no notion of *constraints*. The decoding‑order line (Where‑to‑Unmask, Info‑Gain sampling) patches the decision row only, still committing every step. **Nobody has decoupled propagation from commitment, which is the heart of how real solvers work.** That decoupling is my method.

---

## 4. Problem formulation

A task instance is a natural‑language specification $c$ that *induces* a (possibly implicit) constraint system over an answer canvas $x=(x_1,\dots,x_n)$: a set of constraints $\mathcal{C}(c)=\{C_k\}$ where each $C_k$ couples a subset of positions and a checker returns satisfaction $V(x)\in\{0,1\}$ (a Sudoku validator, an arithmetic‑to‑target evaluator, a graph‑coloring check, a logic‑grid solver, a "does this code satisfy its asserted invariants" checker). $V$ is used for **training and evaluation only**; the deployed model uses no solver.

Define a per‑position **belief** $b_i \in \Delta(\mathcal V)$ (a distribution over candidate tokens) and the canvas belief $B=(b_1,\dots,b_n)$. We want a learned operator $T_\theta$ such that iterating $B \leftarrow T_\theta(B, c)$ **converges to a belief whose mode satisfies $\mathcal C(c)$**, i.e. $T_\theta$ is trained to be a *contraction toward constraint‑consistent fixed points*. A standard DLM is the special case where $T_\theta$ is applied **once per commitment** and beliefs are immediately argmax‑ed and frozen. Relaxation Diffusion lets $T_\theta$ iterate *internally* and commit *lazily*.

**Goal.** Learn $T_\theta$ (initialized from a pretrained DLM) so that, with an *adaptive* number of internal propagation rounds, the model (i) solves harder/denser constraint instances than a vanilla DLM at equal parameter count, (ii) **generalizes to larger instances than seen in training**, and (iii) transfers to *novel constraint types described in natural language* at inference.

---

## 5. Method: Relaxation Diffusion

Three coupled ideas. The first is the core novelty; the others make it trainable.

### 5.1 Decouple propagation from commitment ("belief‑refinement rounds")

Replace the one‑pass "predict → commit lowest‑uncertainty" loop with a two‑timescale loop:

```text
Relaxation Diffusion decoding
  initialize all positions to [MASK]; beliefs B uniform
  repeat (macro = commitment events):
     # ---- inner loop: PROPAGATION, no commitment ----
     repeat r times (r chosen adaptively, see 5.3):
        B <- T_theta(B, c)            # bidirectional pass updates ALL beliefs from current beliefs
                                       # (soft input: feed belief-weighted embeddings, not hard tokens)
        if beliefs are stable (ΔB < ε): break
     # ---- commitment: lazy & decisive ----
     commit positions whose belief is BOTH sharp AND consistent (low marginal constraint-violation);
     leave coupled/contended positions masked for further propagation
  until all committed
```

Key departures from every current DLM:
- **Soft, recirculated beliefs.** The model conditions on *belief‑weighted* (soft) token embeddings of not‑yet‑committed positions, not on `[MASK]`. This lets partial information ("$v_2 \in \{3,7\}$") propagate, rather than collapsing to "unknown." This is the discrete analog of running the diffusion ODE without rounding — and it is what gives the model *candidate sets* to propagate, the thing a solver needs.
- **Internal rounds $r>1$ per commitment.** Hard, densely‑coupled instances get many propagation sweeps before any variable is fixed; easy ones get $r=1$ and behave like a normal DLM. Compute is spent where coupling is high.
- **Lazy, constraint‑aware commitment.** Commit only positions that are sharp *and* not in active contention, the opposite of "commit the most confident token regardless."

### 5.2 Training $T_\theta$ as a propagation operator

Initialize from LLaDA/Dream and fine‑tune with three terms over a curriculum of constraint problems:

$$
\mathcal L = \underbrace{\mathcal L_{\text{MDM}}}_{\text{(1) keep base capability}} \;+\; \lambda_p\,\underbrace{\mathcal L_{\text{prop}}}_{\text{(2) match propagation targets}} \;+\; \lambda_v\,\underbrace{\mathcal L_{\text{solve}}}_{\text{(3) RL on } V}.
$$

- **(1)** standard masked‑token cross‑entropy (so general language ability is preserved; this is a fine‑tune, not from scratch).
- **(2) Propagation supervision.** For synthetic constraint tasks we *have a solver*, so we can generate **intermediate ground‑truth partial assignments** (the candidate sets after $k$ rounds of arc‑consistency / unit propagation) and train the inner operator $T_\theta$ to reproduce that belief‑sharpening trajectory. This directly teaches "what a propagation round should do," rather than hoping it emerges. (Where solvers are unavailable, this term is dropped and we rely on (3).)
- **(3) Solve reward.** RL on terminal correctness $V(x)$, using **SPG's sandwiched‑bound estimator** (this folder) to get a low‑bias policy gradient through the intractable DLM likelihood — reused wholesale, not re‑invented. The reward needs no human labels (the checker computes it).

A **curriculum** scales problem size and constraint density upward, and the propagation‑supervision target (2) is exactly what enables **size generalization**: a model that has learned the *operator* (one consistent propagation step) rather than memorized solutions can apply it for more rounds on larger graphs.

### 5.3 Adaptive propagation budget (where test‑time compute should go)

The number of inner rounds $r$ and the commitment threshold are governed by a learned **contention signal** — the total disagreement between a position's current belief and what its neighbors' beliefs imply (a learned, soft analog of "number of violated constraints"). High contention ⇒ keep propagating, don't commit. This gives a *principled* test‑time scaling axis (more compute on harder instances), distinct from external search (S³, RFG): the extra compute is **internal propagation toward a fixed point**, not sampling more trajectories.

### 5.4 Efficiency

The inner loop costs $r\times$ forward passes per macro‑step, but (i) $r$ is adaptive (≈1 on easy regions), and (ii) it is fully compatible with the efficiency stack in this folder — d²Cache's KV reuse across the (highly similar) inner rounds and DPad's suffix dropout apply directly, so propagation rounds are cheap.

---

## 6. Why this is novel

| Existing line | What it does | Why Relaxation Diffusion is different |
|---|---|---|
| **SPG / d1 / UniGRPO** (RL for DLMs) | Better policy gradient; *observe* the Sudoku gap | We **explain** the gap (propagation) and **amplify** it; SPG's estimator is *reused* as a component, not the contribution. |
| **Decoding‑order / planner** (Where‑to‑Unmask, Info‑Gain, dUltra) | Choose *which* token to commit next, still **commit every step** | We **decouple propagation from commitment** and add **internal belief‑refinement rounds** — a mechanism these do not have. |
| **Self‑correction** (RemeDi, SCOPE) | Re‑mask "low‑quality" committed tokens | We avoid bad commitments *in the first place* via lazy, contention‑aware commitment; correction is a consequence, not the method. |
| **Test‑time search** (S³, RFG, stitching) | Sample/search **many external trajectories** + a reward model | Extra compute is **internal propagation to a fixed point** in *one* trajectory; no external PRM or trajectory bank. |
| **Diffusion for combinatorial optimization** (graphs/TSP, other domains) | Diffusion over a fixed graph structure | Ours operates over **language‑expressed, open‑vocabulary** constraints given in natural language at inference, on a pretrained LM. |

The contribution is **(a)** a mechanistic account of why DLMs are a distinct reasoning substrate (parallel relaxation vs. sequential chaining), grounded in a result from the provided papers; and **(b)** a training/inference paradigm — *propagation/commitment decoupling with soft recirculated beliefs and adaptive internal rounds* — that turns an accidental, unscalable side effect into a deliberate, size‑generalizing capability. Neither is a combination of two prior methods, and neither is a decoding heuristic.

---

## 7. Experimental plan & falsifiable hypotheses

**Base models.** Fine‑tune LLaDA‑8B and Dream‑7B (drop‑in; no from‑scratch training).

**Tasks (with checkers $V$ and solvers for propagation targets).** Sudoku (variable grid size — the size‑generalization stress test), Countdown, graph coloring, logic‑grid puzzles, job‑shop scheduling, SAT/CSP expressed in natural language, and constrained code (functions that must satisfy asserted invariants). Plus GSM8K/MATH to confirm we do not regress weakly‑coupled reasoning.

**Headline hypotheses (each independently falsifiable).**
- **H1 (mechanism).** Intermediate canvases of a *vanilla* DLM already encode partial‑assignment "candidate sets" that sharpen monotonically — i.e. probing recovers solver‑like belief states. *If not, the Propagation Hypothesis is wrong and the framing fails.*
- **H2 (decoupling helps).** At equal parameters and equal *total* compute, adaptive internal propagation rounds beat one‑pass‑per‑commit DLMs, with the **gap growing in constraint density** and approaching zero on weakly‑coupled GSM8K.
- **H3 (size generalization).** A model trained with propagation supervision on $n\le N$ solves $n>N$ instances far better than a vanilla DLM trained on the same data — evidence it learned an *operator*, not solutions.
- **H4 (compute allocation).** The learned contention signal correlates with true #violated‑constraints, and accuracy scales with internal rounds on hard instances while saturating on easy ones (a clean internal test‑time‑scaling law).
- **H5 (NL transfer).** The model solves *novel constraint types* described only in natural language at inference (no such constraint in training), which AR+CoT and vanilla DLMs fail — the payoff that distinguishes "learned a general relaxation engine" from "memorized a puzzle."

**Baselines.** Vanilla LLaDA (low‑confidence remasking), LLaDA+Where‑to‑Unmask, LLaDA+RemeDi, SPG, AR LLaMA‑3‑8B with CoT and with external search/verification.

**Ablations.** Remove soft beliefs (hard `[MASK]` only) → isolates the candidate‑set effect; remove propagation supervision (2) → tests whether RL alone discovers propagation; fix $r=1$ → collapses to a normal DLM; remove lazy commitment → tests the commitment policy.

---

## 8. Risks and limitations

- **Maybe the Sudoku gap is data, not mechanism.** H1 is the make‑or‑break probe; if vanilla canvases don't encode candidate sets, I will report that as a negative result and the framing dies cleanly.
- **Soft‑belief recirculation may be unstable** (a known issue for continuous relaxations of discrete inference). Mitigations: damping/temperature on $T_\theta$, and EBM/MCMC‑style corrected updates (cf. *Reduce, Reuse, Recycle*).
- **Propagation supervision needs solvers**, so it is limited to verifiable/synthesizable domains for the supervised term; term (3) (RL) extends to domains with only a checker. Truly open‑ended language is out of scope for v1.
- **Cost** of inner rounds — argued down via adaptivity + d²Cache/DPad, but must be measured.

---

## 9. Alternative seeds

If you want a different flavor than "reasoning mechanism," here are two other directions I scoped and believe are comparatively under‑mined (each could be expanded into a full proposal):

- **Diffusion features for NLP.** In vision, intermediate diffusion activations are state‑of‑the‑art *representations* (DIFT). No one has asked whether a DLM denoiser's multi‑noise‑level, bidirectional activations form a powerful, multi‑granularity representation for retrieval/parsing/classification — i.e. "is a generative DLM secretly the best encoder?" A clean, mostly‑empirical program with a unify‑gen‑and‑embed payoff.
- **Diffusion‑native architecture.** LLaDA explicitly notes it designed *no* special attention or positional encoding for the diffusion workload (it doesn't even input the noise level). The efficiency papers here (DPad's "scratchpad," d²Cache's "three KV phases") are *evidence* the AR architecture is mismatched to iterative bidirectional refinement. A from‑the‑workload‑up redesign (commitment‑aware positions, mask‑vs‑committed attention routing, time conditioning) is concrete and under‑explored.

Tell me which flavor you want and I'll develop it to the same depth — or push harder on a constraint of your own.

---

## References

**Papers in this folder that this proposal builds on**
- *SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models* — the Sudoku/Countdown gaps that motivate the whole idea; estimator reused in §5.2. https://github.com/facebookresearch/SPG
- *Large Language Diffusion Models (LLaDA)* — bidirectional substrate; reversal‑curse result; "no special architecture" gap (alt‑seed 2).
- *Diffusion LMs Know the Answer Before Decoding (Prophet)* — globally‑stabilizing canvas belief. https://github.com/pixeli99/Prophet
- *DPad* (suffix dropout) & *d²Cache* (dual adaptive caching) — efficiency stack that makes inner rounds cheap (§5.4). https://github.com/Crys-Chen/DPad · https://github.com/Kamichanw/d2Cache

**Field context surfaced during the literature sweep**
- *Where‑to‑Unmask: Ground‑Truth‑Guided Unmasking Order Learning* — https://arxiv.org/abs/2602.09501
- *Improving Sampling for Masked Diffusion Models via Information Gain* — https://arxiv.org/pdf/2602.18176
- *Learning Unmasking Policies for Diffusion Language Models* — https://arxiv.org/pdf/2512.09106
- *Train for the Worst, Plan for the Best: Token Ordering in Masked Diffusions* — https://arxiv.org/html/2502.06768v1
- *Don't Settle Too Early: Self‑Reflective Remasking (RemeDi)* — https://arxiv.org/abs/2509.23653
- *S³: Stratified Scaling Search for Test‑Time in DLMs* — https://arxiv.org/pdf/2604.06260
- *Reduce, Reuse, Recycle: Compositional Generation with EBM Diffusion + MCMC* — stable corrected discrete updates. https://arxiv.org/abs/2302.11552
- *Emergence of a DIFT‑style "diffusion features" literature* (vision; alt‑seed 1) — https://arxiv.org/abs/2306.03881

---

*Grounded in the ten papers in this directory, especially the unexplained constraint‑problem gap in SPG. Designed as a drop‑in fine‑tune that reuses peer‑reviewed components (SPG's estimator, the d²Cache/DPad efficiency stack) so the novel claim — diffusion as learnable constraint propagation — can be tested cheaply and falsified cleanly via H1.*
