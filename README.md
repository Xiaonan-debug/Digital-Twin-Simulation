# Explore the Path, Not the Token
### Coherent Exploration for Reinforcement Learning in Diffusion Language Models

**An RL‑for‑DLM proposal in SPG's methodological style: isolate one factor of the policy‑gradient pipeline that everyone borrows unchanged from autoregressive RL, show it is *structurally broken* for diffusion models, characterize the failure precisely, and replace it with a principled, DLM‑native mechanism — validated on reasoning benchmarks.**

> **The hook.** GRPO‑style RL only learns when a group of rollouts has **reward variance** — i.e., when the policy *explores*. In autoregressive models you dial exploration with one harmless knob: sampling **temperature**. **That knob is broken for diffusion LMs.** Because a DLM commits many tokens *in parallel* from per‑position marginals, raising temperature doesn't produce *diverse valid attempts* — it produces *incoherent* ones (the jointly‑sampled tokens stop agreeing), which earn zero reward and teach nothing. So diffusion RL is silently **exploration‑starved**: too little temperature → identical rollouts → no signal; too much → garbage rollouts → no signal. We give the policy its exploration back by **exploring in trajectory space, not token space** — diversifying *which tokens are committed when*, while keeping every individual commitment cold and coherent — and derive the corrected policy gradient for it.

This is a **different factor** of the same gradient SPG and DACA‑GRPO touch (SPG: the likelihood term; DACA: the credit weights; ours: the exploration distribution), so it **composes** with both.

---

## Table of Contents
1. [Background: RL needs reward variance, reward variance needs exploration](#1-background)
2. [The structural failure: temperature is broken for DLMs](#2-the-structural-failure-temperature-is-broken-for-dlms)
3. [The fix: explore the path, not the token](#3-the-fix-explore-the-path-not-the-token)
4. [The corrected policy gradient](#4-the-corrected-policy-gradient)
5. [Why this is not SPG, DACA, dUltra, or DCoLT](#5-why-this-is-not-spg-daca-dultra-or-dcolt)
6. [Experiments & falsifiable claims](#6-experiments--falsifiable-claims)
7. [Risks](#7-risks)
8. [References](#references)

---

## 1. Background

RL with verifiable rewards (RLVR / GRPO), the workhorse of modern reasoning training and the framework SPG operates in, works as follows for a prompt $q$:

1. Sample a **group** of $G$ rollouts $\{x_1,\dots,x_G\}$ from the policy $\pi_\theta(\cdot\mid q)$.
2. Score them with a verifier, $R(x_i)\in\{0,1\}$ (correct / incorrect).
3. Form the group‑relative advantage $A(x_i) = \dfrac{R(x_i)-\bar R}{\mathrm{std}(R)}$ and update toward high‑advantage rollouts.

The learning signal is **proportional to the spread of $R$ within the group.** If all $G$ rollouts are identical (all right or all wrong), $A\equiv 0$ and the gradient vanishes. The signal therefore depends entirely on the policy's **exploration**: the group must contain *diverse, individually‑coherent attempts* so that some succeed and some fail. This is the engine of RLVR — and it is the part that breaks for diffusion models.

## 2. The structural failure: temperature is broken for DLMs

In an **autoregressive** policy, exploration is dialed by a single scalar — sampling temperature $\tau$ on each token's softmax. It is *harmless* because each token is drawn **conditioned on the actual realized previous tokens**: even at high $\tau$, the sample stays internally consistent (the model conditions on whatever it just sampled). Raising $\tau$ smoothly trades coherence for diversity, and there is a wide usable window. This is why GRPO "just works" for AR.

A **diffusion** policy commits multiple masked positions **in parallel**, each drawn from its own marginal $p_\theta(x_i\mid \text{context})$, *without seeing the other tokens being committed in the same step*. Now raise $\tau$ to explore:

> **Proposition (the exploration bind).** For a diffusion policy, increasing token temperature increases per‑position entropy but, because co‑committed tokens are sampled from a *product of marginals*, it increases the **joint factorization error** super‑linearly: the diverse samples it produces are predominantly **incoherent** (mutually inconsistent tokens), not diverse‑valid. Hence
> - **low $\tau$** → high‑confidence parallel commits → rollouts collapse to near‑duplicates → $\mathrm{std}(R)\!\to\!0$ → **no signal** (starvation);
> - **high $\tau$** → factorization blow‑up → incoherent rollouts → $R\!\equiv\!0$ → **no signal** (garbage).

The usable temperature window for a DLM is therefore *strictly narrower* than for an AR model, and on exactly the hard instances RL most needs to learn — where the base policy is **confidently wrong** — it is often *empty*: the policy is too confident to diversify at low $\tau$ and too fragile to stay coherent at high $\tau$. This is a clean, measurable, DLM‑specific defect that the entire DLM‑RL line (d1, UniGRPO, wd1, SPG, DACA‑GRPO) inherits silently, because they all borrow token‑level sampling from AR and spend their novelty on the *gradient*, not the *exploration*.

The root cause is structural: **AR explores in token space along a fixed generation path; a DLM's coherence lives in the path itself.** So exploration must move to where the structure is.

## 3. The fix: explore the path, not the token

We keep every individual token commitment **cold** (low $\tau$, coherent) and instead inject stochasticity into the **denoising trajectory** — the order and schedule by which positions are committed and revisited. Two coherent‑by‑construction exploration operators:

- **Order perturbation.** Randomize *which* low‑confidence positions are committed at each step (instead of always the top‑$k$ most confident). Different commitment orders steer the rollout into different basins — *different reasoning* — while each committed token is still drawn cold, conditioned on a genuine partial context. Diverse **and** coherent.
- **Stochastic re‑noising (revisitation).** Occasionally re‑mask a committed region and re‑decide it given the now‑richer context. This lets a rollout *escape* the basin the greedy path fell into, producing a genuinely different — but still self‑consistent — final answer.

Intuitively: instead of shaking each token (which shatters coherence), we shake the **route** through the canvas (which preserves it). The result is a group of rollouts that are *individually coherent yet collectively diverse* — restoring the $\mathrm{std}(R)$ that GRPO needs, precisely where token‑temperature cannot.

**Claim:** at any fixed level of per‑sample coherence, path‑space exploration achieves higher *effective rollout diversity* (and thus a stronger advantage signal) than token‑temperature, because it decouples the diversity source from the factorization error.

## 4. The corrected policy gradient

Path‑space exploration changes the sampling distribution, so the gradient must be corrected to stay valid — this is the SPG‑style technical core. Write a rollout as a (final sequence, trajectory) pair $(x,\xi)$ where $\xi$ is the realized commit‑order/revisit schedule drawn from an **exploration kernel** $\rho_\beta(\xi\mid q,\theta)$ (with $\beta$ the exploration strength). The objective is reward of the final sequence, marginal over the nuisance trajectory:

$$
J(\theta)=\mathbb{E}_{\xi\sim\rho_\beta}\;\mathbb{E}_{x\sim\pi_\theta(\cdot\mid q,\xi)}\big[R(x)\big].
$$

The estimator must address two things SPG and DACA do **not**:

1. **An exploration kernel that is differentiated from the policy.** $\rho_\beta$ controls *exploration* (used at rollout time to enrich the group) but the *update* must reinforce the policy's tendency to produce the good final answer **regardless of the path that found it** — otherwise we reinforce a nuisance. We therefore reinforce the **path‑marginalized** likelihood of the final sequence, estimated over the group, rather than the single sampled trajectory's likelihood. (SPG's intractable‑likelihood machinery is *reused here as a black box* to bound the per‑sample term; our contribution is the marginalization over $\xi$.)
2. **An importance correction** $\propto \pi_\theta(\cdot\mid q,\xi)/\rho_\beta$ so that exploring with an off‑policy schedule kernel yields an **unbiased** advantage — analogous in spirit to SPG replacing a biased surrogate with a low‑bias one, but here the bias being removed is the **exploration‑induced** one.

A schedule for $\beta$ (anneal exploration down as the policy sharpens) and a variance analysis (path‑space exploration provably lowers gradient variance versus temperature at matched coherence) complete the method.

## 5. Why this is not SPG, DACA, dUltra, or DCoLT

| Prior line | What it fixes in the gradient | Why this is different |
|---|---|---|
| **SPG** | the **likelihood term** (intractable → sandwiched bound) | We fix the **exploration distribution**; we *reuse* SPG's bound as a component. Composable. |
| **DACA‑GRPO** | the **credit/advantage weights** across the trajectory | We fix **how rollouts are generated** so the group has signal in the first place; DACA assumes signal exists. Composable. |
| **dUltra** | learns one **best unmasking order** to maximize speed/quality (exploit) | We **inject order *stochasticity* to explore** and correct the gradient for it — opposite goal (explore, not exploit), and an unbiased estimator, not a learned planner. |
| **DCoLT** | RL over the unmasking **order as the action** (Plackett‑Luce) to improve reasoning | DCoLT *optimizes* the order as part of the answer; we treat the schedule as an **exploration nuisance to marginalize out**, reinforcing the *path‑independent* final answer. Different objective and different estimator. |

The contribution: a **diagnosis** (token temperature is a structurally broken exploration knob for diffusion policies, with a precise factorization‑error mechanism and a measurable empty‑window failure on hard instances) plus a **fix** (coherent path‑space exploration with an unbiased, path‑marginalized policy gradient). It is orthogonal to, and stackable on, every existing DLM‑RL method.

## 6. Experiments & falsifiable claims

**Base models / setup.** LLaDA‑8B, Dream‑7B; GRPO/SPG trainer; verifiable‑reward reasoning tasks — **GSM8K, MATH500, Countdown, Sudoku** (the SPG suite, for direct comparability).

- **C1 (the bind is real).** Sweep token temperature $\tau$ for a diffusion policy and plot, per instance, group reward‑variance and rollout coherence. *Prediction:* a narrow/empty usable window, with the empty cases concentrated on hard, confidently‑wrong instances. *This is the cheap, first experiment; if the window is wide, the premise is wrong and the project stops here.*
- **C2 (path beats token).** At matched per‑sample coherence, path‑space exploration yields higher effective rollout diversity and higher group reward‑variance than temperature.
- **C3 (it trains better).** Path‑exploration RL improves final accuracy over identical GRPO/SPG runs that explore via temperature — with the largest gains on the hard tasks (Countdown/Sudoku) where the temperature window is emptiest.
- **C4 (composability).** Stacking on SPG (likelihood) and DACA (credit) gives additive gains, confirming it fixes an independent factor.
- **C5 (AR has no such deficit).** The same diagnosis on an AR policy shows a wide temperature window — confirming the deficit is a *diffusion* property, not a generic RL one.

**Ablations.** order‑perturbation vs re‑noising vs both; exploration strength $\beta$ schedule; with/without the importance correction (to show the naive uncorrected version is biased); group size $G$.

## 7. Risks

- **Path‑space exploration might still leave gaps** on instances where *every* coherent path gives the same wrong answer (genuine capability gaps, not exploration gaps) — C1/C3 separate the two; we only claim to fix the *exploration* share.
- **The importance correction could be high‑variance**; mitigations: keep $\rho_\beta$ close to the policy's own sampler, clip ratios (as PPO/GRPO do), anneal $\beta$.
- **Extra rollout cost** from re‑noising — but it is exploration *replacing* failed temperature rollouts, not added on top, and is bounded by $\beta$.

---

## References

**From this folder**
- *SPG: Sandwiched Policy Gradient* — fixes the likelihood term; reused as a component and the methodological template. https://github.com/facebookresearch/SPG
- *Large Language Diffusion Models (LLaDA)* — the parallel‑commitment sampler whose temperature behavior we diagnose.

**Closest neighbors (positioned against in §5)**
- *DACA‑GRPO: Denoising‑Aware Credit Assignment for RL in Diffusion LMs* — https://arxiv.org/abs/2605.16342
- *dUltra: Ultra‑Fast Diffusion LMs via RL* (learned unmasking planner) — https://arxiv.org/html/2512.21446
- *DCoLT: Reinforcing the Diffusion Chain of Lateral Thought* (order‑as‑action RL) — https://arxiv.org/pdf/2505.10446
- *RL for Diffusion LLMs with Entropy‑Guided Step Selection and Stepwise Advantages* — https://arxiv.org/pdf/2603.12554
- *ParallelBench / factorization‑error analyses of parallel decoding* (mechanism behind the temperature bind) — https://arxiv.org/abs/2510.04767

---

*Same DNA as SPG: pick one factor of the policy gradient that the field borrowed from AR without checking, prove it is broken for diffusion models (here: exploration via temperature), characterize the failure, and replace it with a DLM‑native, unbiased fix — demonstrated on the same reasoning suite. The make‑or‑break check (C1) is a temperature sweep on an off‑the‑shelf checkpoint.*
