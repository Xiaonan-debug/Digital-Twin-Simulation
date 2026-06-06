# Flow to a Meaning, Not to the Mean
### Continuous language generation fails because its target is an *average*. We make it commit to one meaning — semantic first, lexical last.

**A new‑paradigm proposal that surpasses ELF in ELF's own frame. ELF's thesis was "continuous DLMs were held back by a *habit*, not a limitation" — the habit being per‑step token anchoring, which it removed. We name the *deeper* habit ELF kept: the flow is trained by MSE, so its target is the **conditional mean** of all destinations. For images the mean is a benign blur; for language the mean of two meanings is a *non‑meaning*. The flow spends its whole early budget aiming at a point between "the cat sat" and "a dog ran" — a place no sentence lives. We replace the mean target with a **decisive, mode‑committing target** whose commitment is *scheduled* — meaning's mode locks first, words last — turning the velocity field from an averager into a generator of `meaning → structure → text`.**

> **The one‑sentence diagnosis.** Flow matching learns $v_\theta(z,t)\!\approx\!\mathbb E[\,e(x)-\varepsilon \mid z_t=z\,]$ — a **conditional expectation**. Expectation over a *unimodal‑ish* image posterior is a slightly blurry image, still on the manifold. Expectation over a *wildly multimodal* language posterior is the **centroid of unrelated sentences**, which is off the manifold of meaning. Continuous DLMs have been flowing toward centroids of meaning this whole time. That is why a half‑finished continuous DLM state is mush, and why the field needed a hard token read‑out bolted onto the end. **The target, not the space and not the corruption, is the bug.**

---

## Table of Contents
1. [The habit ELF kept](#1-the-habit-elf-kept)
2. [Why the mean is benign for images and fatal for language](#2-why-the-mean-is-benign-for-images-and-fatal-for-language)
3. [What we actually want: the mode, committed in semantic order](#3-what-we-actually-want)
4. [The method: Decisive Flows](#4-the-method-decisive-flows)
5. [The central challenge, stated honestly](#5-the-central-challenge-stated-honestly)
6. [Why this is a new paradigm, not ELF + X](#6-why-this-is-a-new-paradigm)
7. [Experiments & falsifiable claims](#7-experiments--falsifiable-claims)
8. [Risks](#8-risks)
9. [References](#references)

---

## 1. The habit ELF kept

A continuous DLM has three design choices, not one:

| Choice | Image diffusion convention | ELF | LangFlow | **left unexamined by all** |
|---|---|---|---|---|
| the **space** | pixels / VAE latent | learned embedding | embedding | — |
| the **forward** corruption | Gaussian | Gaussian | Gaussian (+ learned *scalar* schedule) | — |
| the **regression target** | conditional **mean** velocity (MSE) | conditional **mean** (MSE) | conditional mean (Bregman/CE) | **← this one** |

ELF's contribution was to stop forcing the model to round to a token at every step ("the habit"), giving the flow "maximal flexibility." LangFlow's was a learnable scalar noise schedule and self‑conditioning. **Neither touched what the velocity field is *regressed toward*.** And that target is, by the definition of the flow‑matching loss, a **conditional expectation** — an average over every clean destination consistent with the current noisy state.

For a continuous, near‑unimodal modality this is exactly right; it is the whole reason diffusion works. For language it is the quiet catastrophe, because the set of destinations consistent with a high‑noise embedding is not a tight cluster — it is *every sentence in the language*, and their average is a vector that decodes to nothing.

## 2. Why the mean is benign for images and fatal for language

Take a noisy state $z_t$ at high $t$ (mostly noise). The flow‑matching target is built from the **posterior over clean data** $p(x\mid z_t)$.

- **Images.** $p(x\mid z_t)$ at high $t$ is broad but *locally connected*: the consistent images are blurry variations of one another. Their mean is a **blurry but valid image** — on the manifold, coarse‑to‑fine for free. Averaging = low‑pass filtering = the coarse‑structure‑first schedule image diffusion gets gratis.
- **Language.** $p(x\mid z_t)$ at high $t$ is **multimodal across *meanings*** — "the cat sat on the mat," "stocks fell sharply," "je t'aime" are all roughly equally consistent with near‑pure noise. Their embedding‑space average is the centroid of unrelated meanings: **off‑manifold, decodes to nothing.** Averaging here is not low‑pass filtering; it is *destroying* the very semantic decision the essay says must be made first.

Decompose the destination embedding into a **semantic** part and a **lexical** part, $e(x)=e_{\text{sem}}\oplus e_{\text{lex}}$ (semantic = a frozen sentence/structure encoder's subspace; lexical = the surface residual). The posterior spread splits the same way:

$$
\underbrace{\operatorname{Var}[\,e_{\text{sem}}\mid z_t\,]}_{u_{\text{sem}}(z_t)} \;+\; \underbrace{\operatorname{Var}[\,e_{\text{lex}}\mid z_t\,]}_{u_{\text{lex}}(z_t)} .
$$

This is **exactly the essay's $u=u_{\text{sem}}+u_{\text{lex}}$.** The pathology, stated precisely:

> At high $t$, $u_{\text{sem}}\!\gg\!u_{\text{lex}}$ — most of the posterior spread is *across meanings*. The MSE target averages over that spread, so the early flow's velocity is dominated by **semantic averaging**: it pulls toward the mean of all meanings instead of *choosing* one. The essay says resolve $u_{\text{sem}}$ **first**; the mean‑velocity flow resolves it **never** — it dilutes it into a centroid and hopes the terminal read‑out invents a word. That terminal read‑out (ELF's shared‑weight snap at $t\!=\!1$) is a *symptom*: you only need a hard de‑blurring step at the end if the flow spent the whole trajectory aiming between meanings.

This reframes the "stop‑halfway = garbled string" diagnostic mechanistically: the halfway state is garbled **because it is a literal average of incompatible sentences**, not because the space or the corruption is wrong.

## 3. What we actually want

Not the mean of the posterior — **a mode**, committed in the right order:

1. **Mode, not mean.** The velocity should point at *one* consistent destination, not the centroid of all of them. (A blurry valid sentence is impossible; a *committed* one is the only kind that exists.)
2. **Commit semantics before lexis.** *Which* meaning is decided at high $t$ (low information, cheap, paraphrase‑invariant); *which words* render it is decided at low $t$. So the commitment is a **schedule over subspaces**, not a single event.
3. **Decoupled clocks (the essay's headline).** The **noise schedule** (how much $\varepsilon$ remains, indexed by $t$) and the **commitment schedule** (how sharply the flow has chosen a mode, per subspace) are *different objects*. At one noise level the flow can be fully committed in meaning and fully uncommitted in wording. *Noise schedule ≠ generation schedule*, made literal.

A model that does (1)–(3) generates `pick a meaning → arrange it → spell it`, and every halfway state is a **coherent committed gist**, not an average — the stop‑halfway diagnostic satisfied by construction, with **no terminal read‑out crutch**.

## 4. The method: Decisive Flows

Keep ELF's space and Gaussian forward (we are not re‑litigating those — that's [PRISM]/[Semantic Corruption Flows] territory). Change **only the target and its schedule.** Three components.

**(a) Space (ELF‑minimal).** $z_1=e(x)$ in a learned embedding space; $z_0=\varepsilon\sim\mathcal N(0,I)$; straight interpolant $z_t=(1-t)\varepsilon+t\,e(x)$. A frozen semantic encoder defines an orthogonal split $P_{\text{sem}},P_{\text{lex}}$ of that space (sentence‑embedding directions vs residual).

**(b) A mode‑committing target via responsibilities (the core).** Standard flow matching regresses to $\mathbb E[e-\varepsilon\mid z_t]$. We instead regress to a **responsibility‑weighted target that is driven toward a single mode**. Given $z_t$ and a set of candidate destinations $\{c_k\}$ — minibatch targets, or a learned **meaning codebook** of prototypes — define responsibilities and the target

$$
r_k(z_t;\beta_t,\,M_t)=\frac{\exp\!\big(-\beta_t\,\lVert M_t(z_t-c_k)\rVert^2\big)}{\sum_j \exp\!\big(-\beta_t\,\lVert M_t(z_t-c_j)\rVert^2\big)},
\qquad
\tilde u_t(z_t)=\sum_k r_k\,(c_k-\varepsilon).
$$

- $\beta_t$ is an **inverse temperature**: $\beta_t\!\to\!0$ recovers the soft average (≈ELF's mean); $\beta_t\!\to\!\infty$ gives a **hard mode** (argmax destination). We *anneal $\beta_t$ up as the flow proceeds*, so the flow **decides**.
- $M_t$ is a **subspace mask that schedules *what kind* of mode is committed when.** Early ($t$ small in the reverse): $M_t=P_{\text{sem}}$ — responsibilities are computed in the *semantic* subspace, so the flow commits to a **meaning cluster** while staying agnostic about words. Late: $M_t\!\to\!I$ — responsibilities sharpen in the full space, committing **words**. This is $u_{\text{sem}}$‑before‑$u_{\text{lex}}$ written directly into the loss.

Train $v_\theta$ with $\mathbb E_t\,\lVert v_\theta(z_t,t)-\tilde u_t(z_t)\rVert^2$. When $\beta_t\!=\!0,M_t\!=\!I$ this **is** ELF — Decisive Flows contains ELF as its no‑commitment limit, so the downside is bounded and the ablation is one knob.

**(c) Two clocks, explicitly.** The noise level $t$ and the commitment pair $(\beta_t,M_t)$ are separate schedules. The contribution is not "add a temperature" — it is the claim that **the commitment schedule should be semantic‑first and is a first‑class design axis the field has been holding fixed at $\beta\!=\!0$.** We learn $(\beta_t,M_t)$ under a meaning‑first prior (semantic commitment time < lexical commitment time), the way LangFlow learned the *noise* schedule — but here the schedule governs *which uncertainty is resolved*, which is the essay's actual subject.

**What this buys.** The flow never aims at a centroid of meanings: at every step it is pointed at one chosen destination, coarsely (a meaning) then finely (a sentence). The terminal hard read‑out becomes unnecessary because the trajectory was decisive all along — decoding $z_t$ at any $t$ yields **a real, progressively‑specified sentence**: topic → gist → draft → text.

## 5. The central challenge, stated honestly

The hard, *interesting* part — the reason this is a research program, not a temperature knob — is **committing without collapsing.** A naively sharpened target invites two failure modes, and handling them is the work:

- **Mode collapse / mean‑seeking tension.** Anneal $\beta_t$ too fast and the flow locks onto a single global mode (low diversity); too slow and you recover the averaging blur. The defense is the *subspace schedule*: commit in the **low‑dimensional semantic** subspace first (few, well‑separated meaning modes → safe to commit early) and only later in the high‑dimensional lexical residual (commit late, gently). Diversity comes from *which* meaning mode is entered (set by the prior / early stochasticity), not from indecision.
- **Defining "semantic" without circularity.** $P_{\text{sem}}$ is grounded *externally* by a frozen encoder, so the model cannot define meaning to suit itself — exactly as the essay demands we learn *where* meaning becomes text rather than asserting it. Sensitivity to this choice is an explicit ablation.
- **The codebook vs. minibatch choice.** Responsibilities over a learned meaning codebook give a clean, reusable set of modes but risk under‑coverage; minibatch responsibilities are coverage‑complete but noisy. Which to use, and whether the codebook should be *generated first by an inner fast flow*, is an open sub‑question (and a clean place for follow‑up, not a dependency).

Whether a scheduled, mode‑committing target can realize "decide meaning first, words last" for natural language is precisely the essay's open question — and C1 tests it cheaply, before any scaling.

## 6. Why this is a new paradigm

| | ELF | LangFlow | OT‑/multisample Flow Matching | Simplex / discrete DLMs | **Decisive Flows** |
|---|---|---|---|---|---|
| What's changed | removed per‑step CE anchoring | learned **scalar** noise schedule + self‑cond. | better noise↔data **coupling** (lower transport cost) | model categorical posterior directly | **the regression *target*: mean → scheduled mode** |
| Velocity target | conditional **mean** | conditional mean (Bregman) | conditional mean (straighter paths) | n/a (categorical) | **responsibility‑weighted *mode*, annealed** |
| Resolves $u_{\text{sem}}$ before $u_{\text{lex}}$? | no | no | no | no (per‑token, order‑agnostic) | **yes — by the subspace schedule $M_t$** |
| Halfway state | centroid of meanings (mush) | mush | mush | sentence with holes | **a committed coarse meaning (real text)** |
| Needs terminal hard read‑out? | yes (signature move) | yes | yes | n/a | **no — decisive throughout** |

This is **not ELF + a trick.** ELF's whole story is "remove the habit that cripples continuous DLMs." We show the habit it removed was the *second* one; the *first* — regressing to the mean of a multimodal language posterior — is still there, and is the one the essay's diagnostics actually indict. Removing it changes the object the flow optimizes, not the space (PRISM's axis), not the corruption (Semantic Corruption Flows' axis), and not the sampler. And it is the opposite move from OT‑CFM: OT‑CFM keeps the mean target and makes the *paths* straighter to reach it faster; we keep the paths and change the *target* so the destination is a meaning rather than an average. The two are orthogonal and composable, which is itself evidence they are different axes.

It also pays a conceptual debt: LangFlow reported that self‑conditioning helps continuous DLMs "with effects substantially different from discrete diffusion" and could not say why. Our frame **explains it** — self‑conditioning is a weak, accidental symmetry‑breaker that nudges the flow off the centroid toward a mode. Decisive Flows makes that deliberate, semantic, and scheduled.

## 7. Experiments & falsifiable claims

**Setup (ELF protocol).** Small scale first (105–170M), OWT/LM1B; report Gen‑PPL and step‑efficiency; then summarization/MT (ELF's own downstream) and planning‑heavy tasks where meaning‑order matters.

- **C1 — the pathology is real and is semantic (make‑or‑break, runs on a *frozen* ELF, no training).** Sample states $z_t$ along ELF trajectories; estimate the posterior spread by collecting minibatch destinations near each state and measuring $u_{\text{sem}}(z_t)$ vs $u_{\text{lex}}(z_t)$. **Prediction:** at high $t$, $u_{\text{sem}}\!\gg\!u_{\text{lex}}$, and ELF's learned velocity sits near the *centroid* of multiple meaning clusters (high angular error to every individual mode). *If the posterior is not semantically multimodal, or ELF already tracks a single mode, the diagnosis is wrong and the paradigm dies cheaply.*
- **C2 — committing fixes it.** A controlled toy: 2 meanings × 3 paraphrases each. ELF's flow lands between the two meaning clusters (decodes to neither); Decisive Flows enters one cluster early (semantic commitment) and one paraphrase late (lexical). Quantify "between‑meaning" residence time.
- **C3 — the schedule is semantic‑first.** Probing $z_t$ recovers the final *meaning* (frozen sentence‑embedding) far earlier in $t$ than it recovers exact *tokens* — and earlier than for ELF — confirming $u_{\text{sem}}$ resolved before $u_{\text{lex}}$.
- **C4 — efficiency, ELF's currency.** Because the costly semantic decision is made decisively early instead of being deferred to a terminal de‑blur, target Gen‑PPL is reached in **fewer flow steps**, and the dependence on the terminal hard read‑out weakens (ablate the read‑out: ELF degrades sharply, Decisive Flows much less).
- **C5 — it helps where meaning‑order matters.** Largest wins on long‑form coherence, summarization faithfulness, and controllable generation by intervening on the *early semantic* state (paraphrase‑robust control), vs brittle token/embedding edits in ELF.

**Ablations.** $\beta_t\!\equiv\!0$ (recovers ELF — isolates the entire contribution); fix $M_t\!=\!I$ (commit, but not semantic‑first — isolates the *schedule* from the *commitment*); reverse the schedule (lexical‑first — should hurt, proving order matters); codebook vs minibatch responsibilities; source of $P_{\text{sem}}$ (sentence‑encoder vs syntactic vs answer‑bearing).

## 8. Risks

- **Mode collapse / diversity loss** — the central risk (§5); defended by committing only in the low‑dim semantic subspace early and keeping lexical commitment soft, plus standard entropy/diversity monitoring. C2 watches for it directly.
- **$P_{\text{sem}}$ externally defines "semantic"** — a modeling choice; tested for sensitivity, with fully‑emergent semantic subspaces as the harder follow‑up.
- **Responsibility estimation cost** — minibatch responsibilities add a softmax over candidates per step; the codebook variant amortizes it, and it composes with the d²Cache/efficiency stack.
- **Worst case** the annealing buys nothing and we fall back to $\beta\!=\!0$ — i.e., we recover an ELF‑class model. Bounded downside, because ELF is the no‑commitment limit of our objective.

---

## References

**From this folder (the lessons taken and surpassed)**
- *ELF: Embedded Language Flows* — continuous embedding‑space Flow Matching; removed per‑step token anchoring; we remove the *mean‑velocity* assumption it kept. https://arxiv.org/abs/2605.10938
- *LangFlow* — learnable *scalar* noise schedule + self‑conditioning; we learn a *commitment* schedule and explain why self‑conditioning helped. https://arxiv.org/abs/2604.11748
- *Continuous Latent Diffusion Language Model (Cola)* — compressed semantic latent; orthogonal (space, not target). https://arxiv.org/abs/2605.06548

**Sibling proposals in this folder (distinct axes — not duplicated)**
- *PRISM* — changes the *noise variance per spectral mode*. *Semantic Corruption Flows* — changes the *forward corruption operator*. **Decisive Flows changes the *regression target* and its commitment schedule** — a third, independent axis, composable with both.

**Field context surfaced in the literature sweep**
- *Mean‑seeking / posterior‑mean prediction in diffusion* — the conditional‑expectation target underlying flow matching; benign blur for images, off‑manifold for multimodal language.
- *OT‑/multisample Flow Matching* — straighter paths to the *same* mean target; orthogonal and composable (positioned against in §6).
- *DDOT: Flexible‑length Text Infilling* (arXiv 2506.13579) — *discrete* diffusion that denoises token positions; different axis (structure variable) and discrete, not a scheduled‑mode target.

**The essay (provided as screenshots)**
- *noise schedule ≠ generation schedule*; the *stop‑halfway* diagnostic; $u=u_{\text{sem}}+u_{\text{lex}}$, resolve semantic before lexical; `meaning → structure → text`; *"learn **where** meaning becomes text."*

---

*Image diffusion got its coarse‑to‑fine schedule for free because, for images, the mean of the posterior is a valid blurry picture. Language has no such luck: the mean of two meanings is a non‑meaning, and continuous DLMs have been flowing toward it the whole time — patching the damage with a hard token read‑out at the very end. Decisive Flows stops flowing toward the average and starts flowing toward a chosen meaning, committing semantics before lexis. That makes the **regression target** — untouched by ELF, LangFlow, and every continuous DLM — the object of study, and turns the velocity field from an averager into the meaning‑first generator the essay asks for. C1 decides it on a frozen checkpoint, before a single training step.*
