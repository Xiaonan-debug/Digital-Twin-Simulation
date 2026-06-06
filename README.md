# PRISM — Meaning Lives in the Slow Modes
### The generation schedule is *already* spectral. We just pointed it at the wrong modes.

**A new‑paradigm proposal in ELF's spirit — continuous space, one clean idea, falsifiable on a small model. ELF taught us: stay in continuous space, discretize only at the end. The essay taught us: a DLM must learn `meaning → structure → text`, and above all *where* meaning becomes text. The move no one makes: stop hand‑designing a generation *schedule* at all. Every continuous diffusion model already has one — its own *spectral bias* resolves high‑variance directions first — and in embedding space that order is surface‑first. So don't add machinery. Re‑tune the space so its variance spectrum *is* the semantic spectrum, and the schedule `meaning → structure → text` falls out of physics you already have.**

> **The observation that decides everything.** A recent result makes diffusion's generation order exact, not folklore: a mode with data‑covariance eigenvalue $\lambda_k$ emerges with timescale $\tau_k \propto \lambda_k^{-1}$ — **high‑variance modes are generated first** (Wang et al., *Spectral Bias in Diffusion*, 2026). For **image pixels** the top‑variance modes happen to be coarse global structure, so denoising is meaning‑/structure‑first *for free* — the reason image diffusion "just works." For **language embeddings** (ELF's space) the top‑variance modes are *token identity and surface frequency*. So the very same physics generates **surface‑first**: exactly the `noise → tokens` failure the essay names. **The schedule was never missing. It was misaligned.**

**The thesis.** The essay asks for two things: a *structured generation space* (结构化生成空间) and a *semantic generation schedule* (语义化生成调度). PRISM gets both from one act. We learn a latent space whose **principal axes of variance are the axes of meaning** — concretely, a space where *variance between different meanings* is packed into a few top modes and *variance from paraphrase / surface choice* is pushed into many small modes. Then we run plain ELF‑style Flow Matching, unchanged. Because $\tau_k \propto \lambda_k^{-1}$, the now‑high‑variance **semantic** modes resolve first and the low‑variance **lexical** modes resolve last. The schedule is not engineered; it is *induced* by the geometry. And the essay's hardest question — *where* meaning becomes text — gets a measurable answer: **the elbow in the variance spectrum**, the band edge where modes stop being paraphrase‑invariant and start being surface‑specific.

> A prism doesn't add light. It reveals the order that was always inside the beam. PRISM doesn't add a schedule. It reveals — and re‑tunes — the spectral order already inside the flow.

---

## Table of Contents
1. [The unexamined fact: diffusion already has a schedule](#1-the-unexamined-fact-diffusion-already-has-a-schedule)
2. [Why it's right for images and wrong for language](#2-why-its-right-for-images-and-wrong-for-language)
3. [The reframing: align the variance spectrum with the meaning spectrum](#3-the-reframing)
4. [The method: PRISM](#4-the-method-prism)
5. [Where meaning becomes text, made measurable](#5-where-meaning-becomes-text-made-measurable)
6. [Why this is a new paradigm, not ELF/LangFlow/Cola + X](#6-why-this-is-a-new-paradigm)
7. [Experiments & falsifiable claims](#7-experiments--falsifiable-claims)
8. [Risks](#8-risks)
9. [References](#references)

---

## 1. The unexamined fact: diffusion already has a schedule

A diffusion / flow model has a forward corruption $q(z_t\mid z_0)$ and a reverse generator $p_\theta$. The field pours innovation into the reverse half and the *space*; the folder's own *Semantic Corruption Flows* proposal pours it into learning the forward half. **Both treat the generation schedule — the order in which information appears — as something you must add.** It is not. It is already there, and it is a *spectrum*.

Decompose the data in the eigenbasis of its covariance $\Sigma=\mathrm{Cov}(z_0)=\sum_k \lambda_k\, e_k e_k^\top$. A linear‑denoiser analysis — and, empirically, deep ones — gives a clean law for *when* each coordinate $\langle z_0, e_k\rangle$ is fixed along the reverse trajectory:

$$
\boxed{\;\tau_k \;\propto\; \lambda_k^{-1}\;}\qquad\text{(emergence time of mode }k\text{)}
$$

High‑variance modes ($\lambda_k$ large) are settled **early**; low‑variance modes are settled **late**. This is the *spectral bias / spectral‑law ordering of mode convergence* (Wang et al., 2026; see also analyses of diffusion learning dynamics). It is not a knob. It is what diffusion *does*. **Every continuous DLM in this folder — ELF, LangFlow, Cola — inherits this schedule whether it knows it or not.** The only freedom is: *which directions in the latent get the large $\lambda_k$?* That single choice silently decides what the model generates first.

## 2. Why it's right for images and wrong for language

The eigenmodes $e_k$ are properties of the **space**, and the spectrum $\lambda_k$ is a property of the **data in that space**. The schedule $\tau_k\propto\lambda_k^{-1}$ is therefore only as good as the alignment between "high variance" and "what should be decided first."

| | top‑variance modes $e_{k}$ (large $\lambda_k$) are… | so the *free* schedule generates… | half‑finished state is… |
|---|---|---|---|
| **Image, pixel space** | low spatial frequency: coarse global structure | structure → detail, *for free* | a blurry but **coherent** image |
| **Language, ELF embedding space** | token identity, frequency, position: **surface** | **surface → meaning** (backwards!) | a fluent‑looking but **meaning‑less** draft |
| **Language, PRISM space (ours)** | **between‑meaning** variation: intent, topic, relations | **meaning → structure → text** | a coherent **gist / outline** |

Image diffusion's "magic" was never the Gaussian and never a clever schedule. It was a lucky coordinate system: pixel variance is dominated by coarse structure, so $\tau_k\propto\lambda_k^{-1}$ *happens* to mean coarse‑first. Language hands us the opposite coordinates. In token‑embedding space the dominant variance is *which tokens appear* — surface. So ELF's own spectral bias drives it to commit surface statistics first and backfill meaning, the precise inversion the essay diagnoses as `noise → tokens`. **The bug is not the schedule and not the denoiser. It is that, in language, variance ≠ significance.** Fix the coordinates and the schedule fixes itself.

## 3. The reframing

The essay's uncertainty decomposition $u(x)=u_{\text{sem}}(x)+u_{\text{lex}}(x)$ becomes, in this lens, a statement about *where the variance lives*. Use the law of total covariance over a meaning variable $c$ (a content/intent) and its surface realizations $x\sim p(x\mid c)$:

$$
\Sigma \;=\; \underbrace{\mathrm{Cov}_c\big[\mathbb{E}[z\mid c]\big]}_{\textstyle \Sigma_{\text{between}}\;=\;\text{meaning } (u_{\text{sem}})}
\;+\; \underbrace{\mathbb{E}_c\big[\mathrm{Cov}[z\mid c]\big]}_{\textstyle \Sigma_{\text{within}}\;=\;\text{surface } (u_{\text{lex}})}.
$$

$\Sigma_{\text{between}}$ is variation *across meanings* (paraphrases share it); $\Sigma_{\text{within}}$ is variation *across paraphrases of one meaning* (pure surface). The essay's ideal — **resolve $u_{\text{sem}}$ before $u_{\text{lex}}$** — is, given $\tau_k\propto\lambda_k^{-1}$, *exactly* the geometric condition:

$$
\boxed{\;\text{top eigenmodes of }\Sigma \;=\; \text{range of }\Sigma_{\text{between}};\qquad \Sigma_{\text{within}}\ \text{confined to the small‑}\lambda\ \text{tail.}\;}
$$

In words: **make meaning the high‑variance directions and surface the low‑variance directions.** Do that, and diffusion's own physics generates meaning first — no learned corruption operator, no new sampler, no schedule network. This is the whole idea. It is a *generative* cousin of Fisher's between/within decomposition (LDA), but where LDA seeks a discriminative projection, PRISM seeks a **generative latent whose variance spectrum doubles as a semantic significance spectrum**, so that the *order of generation* — not a classifier boundary — is what gets organized.

## 4. The method: PRISM

Three components. The reverse generator is deliberately ELF‑plain; the novelty is entirely in *shaping the spectrum of the space*, and in proving the schedule that results is semantic.

**(a) Continuous space (ELF's lesson, kept minimal).** Encoder $\phi:\mathcal X\to\mathcal Z$, decoder $\psi:\mathcal Z\to\mathcal X$. $z_0=\phi(x)$; tokens read out by $\psi$ at $t\to0$. We add *no* discretization until the end.

**(b) Spectral shaping of $\phi$ — the heart.** A *meaning signal* groups surface variants: paraphrase pairs from back‑translation / dropout‑augmentation, or soft grouping by a frozen sentence encoder $s(\cdot)$ (used only to *define* "same meaning," never decoded from). We train $\phi,\psi$ so that the latent covariance is **spectrally sorted by meaning**:

$$
\mathcal L \;=\; \underbrace{\mathcal L_{\text{recon}}(\phi,\psi)}_{z_0\text{ still encodes all of }x}
\;+\;\lambda_1\underbrace{\mathcal L_{\text{between}\uparrow}}_{\text{pack meaning into few high-}\lambda\text{ modes}}
\;+\;\lambda_2\underbrace{\mathcal L_{\text{within}\downarrow}}_{\text{push surface into the low-}\lambda\text{ tail}}
\;+\;\lambda_3\underbrace{\mathcal L_{\text{order}}}_{\text{nested: significance monotone in }k}.
$$

- $\mathcal L_{\text{between}\uparrow}$: maximize variance of $\mathbb E[z\mid c]$ along the **top‑$K$** axes — paraphrases of *different* contents must spread out there. Meaning becomes low‑dimensional and high‑variance.
- $\mathcal L_{\text{within}\downarrow}$: minimize paraphrase variance in the top‑$K$ axes (same meaning ⇒ same top‑$K$ coordinates) and let it live only in the small‑$\lambda$ tail. Surface becomes high‑dimensional and low‑variance.
- $\mathcal L_{\text{order}}$: a *nested* / sequential‑subspace constraint (à la nested dropout) so that significance is **monotone in $k$** — there is a single ordered spectrum, not an unordered split. This is what makes the schedule a smooth `meaning → structure → text`, and what makes the band edge well‑defined.

Crucially, this **does not learn a forward operator** and cannot collapse to a degenerate corruption (the failure mode the folder's SCF proposal must defend against): the forward process stays a fixed, simple Flow‑Matching path. We only re‑sort the coordinates of $\mathcal Z$. The fixed points are pinned by an *externally defined* equivalence (paraphrase = same meaning), so "semantic" is grounded, not circular.

**(c) Plain Flow Matching, unchanged (ELF).** Train the reverse velocity $v_\theta$ with standard conditional Flow Matching in $\mathcal Z$. *Nothing about the sampler is modified.* The meaning‑first schedule is now a free consequence of $\tau_k\propto\lambda_k^{-1}$ acting on a spectrum we engineered.

**(d) Optional amplifier — a *spectral* schedule.** Free spectral bias gives the *ordering*; if we want to *sharpen* the separation (lock meaning hard before surface unfreezes), give each mode its own noise rate $\sigma_k(t)$ tied to its band — a **per‑mode schedule** that generalizes LangFlow's single *scalar* learnable schedule to a *spectrum* of schedules, now derived from $\lambda_k$ rather than hand‑learned. This is strictly optional: the minimal, most beautiful PRISM is geometry + untouched physics.

```text
PRISM, end to end
  encode:     z0 = φ(x)                      # latent whose top modes = meaning
  (train φ,ψ so Cov(z0) is spectrally sorted: between-meaning ↑ top, within-meaning ↓ tail)
  generate:   z1 ~ prior;  integrate dz/dt = v_θ(z,t)   # plain Flow Matching, no schedule tricks
              └─ spectral bias τ_k ∝ 1/λ_k  ⇒  top (semantic) modes fix first, tail (lexical) last
  read out:   x = ψ(z0)                      # discretize only at t→0
  diagnostic: stop at t>0 ⇒ only top modes resolved ⇒ ψ(z_t) = a coherent GIST
```

## 5. Where meaning becomes text, made measurable

The essay's deepest, least‑answered demand — *"它首先应学习意义在哪里能够变成文本"* (first learn **where** meaning can become text) — becomes a number. Plot the per‑mode **between‑meaning fraction** $\rho_k=\dfrac{e_k^\top\Sigma_{\text{between}}e_k}{e_k^\top\Sigma e_k}$ against $k$. PRISM predicts a sigmoid with a sharp **elbow $K^\star$**:

- modes $k\le K^\star$: $\rho_k\approx 1$ — **paraphrase‑invariant, semantic** (the meaning band; emerges first);
- modes $k> K^\star$: $\rho_k\approx 0$ — **surface‑specific, lexical** (the form band; emerges last).

$K^\star$ — the band edge — **is** "where meaning becomes text," reported as a discovered property of language rather than a hyperparameter. The essay's `meaning → structure → text` predicts not one elbow but a *three‑band* spectrum: a top band (intent/topic), a middle band (discourse/syntactic skeleton — "structure"), a high band (exact words). We test for exactly this banding (C2). And the "stop‑halfway" diagnostic is true *by construction*: halting the flow at $t>0$ leaves only the top band resolved, so $\psi(z_t)$ decodes a coherent coarser text — a gist, then an outline, then a draft — as $t\to0$.

## 6. Why this is a new paradigm

| | ELF | LangFlow | Cola | *Semantic Corruption Flows* (this folder) | **PRISM** |
|---|---|---|---|---|---|
| Generation **space** | raw embedding | embedding | compressed VAE latent | ELF‑like | **embedding re‑tuned so variance‑axes = meaning‑axes** |
| **Schedule** comes from | implicit spectral bias (ignored) | a learned **scalar** noise schedule | implicit, in latent | a **learned forward operator** | **implicit spectral bias on an engineered spectrum** |
| What is **learned** | reverse + space | reverse + scalar schedule | reverse + VAE | reverse + forward corruption | **reverse + the *eigenbasis of meaning*** |
| Meaning‑first? | no — surface‑first | no — one global clock | partial (compression/block‑causal) | by an added operator | **yes — from physics, once coordinates are right** |
| "Where meaning becomes text" | — | — | implicit in bottleneck | a trained trajectory | **a measured spectral band edge $K^\star$** |
| Forward process | fixed Gaussian | fixed Gaussian + scalar sched. | fixed Gaussian | **replaced by learned operator** | **untouched** |

The contrast that matters most is with the folder's own essay‑driven proposal. *Semantic Corruption Flows* answers the essay by **learning the forward corruption** so meaning dissolves gist‑last — powerful, but it must fight operator collapse and add probe/fluency losses to keep the corruption honest. **PRISM answers the opposite way: leave the physics alone and move the coordinates.** It exploits a *law the field just discovered* (spectral bias) instead of building machinery to imitate that law. No operator to collapse; the only learned object is an *orthonormal basis sorted by an externally grounded notion of meaning*. It is the **structured generation space** half of the essay's prescription (结构化生成空间), executed so completely that the **semantic schedule** half comes for free. It is not ELF+X, not LangFlow+X, not Cola+X: it is the claim that *the generation schedule was always a property of the space's spectrum*, and language simply had the wrong spectrum.

> **The one‑sentence thesis:** *meaning is the slow mode* — and once you make meaning the high‑variance direction, diffusion's own spectral bias generates it first.

## 7. Experiments & falsifiable claims

**Setup (ELF protocol).** Small scale first (≈105–170M), OWT / LM1B; report Gen‑PPL and few‑step efficiency; then planning‑heavy and controllable tasks. Meaning signal from back‑translation + a frozen sentence encoder for grouping.

- **C1 — the spectrum is semantic (make‑or‑break, frozen‑model cheap).** After spectral shaping, plot $\rho_k$ vs $k$. **Claim:** a clear monotone elbow $K^\star$ — top modes paraphrase‑invariant, tail modes surface‑specific. *And* truncate $z_0$ to its top‑$K$ modes and decode: meaning must be preserved while wording degrades gracefully as $K\downarrow$ (full sentence → summary → headline → topic). *If $\rho_k$ is flat (no semantic ordering) or truncation destroys meaning before surface, the paradigm fails — and this runs on the encoder alone, before any flow training.*
- **C2 — three‑band structure.** The spectrum shows the essay's `meaning → structure → text` as *bands*: a top band predicting intent/topic, a middle band predicting discourse/syntactic skeleton, a high band predicting exact tokens. Probes recover this banding; ELF's raw‑embedding spectrum does not.
- **C3 — it generates meaning‑first (the payoff).** During *unmodified* Flow‑Matching sampling, the top‑band coordinates of $z_t$ stabilize early and the tail late (measure per‑mode settling time; compare to $\tau_k\propto\lambda_k^{-1}$). For ELF the order is reversed (surface settles first). Intervening on early states changes meaning paraphrase‑robustly; intervening late changes only wording.
- **C4 — efficiency (ELF's currency).** Because the few high‑variance semantic decisions are made first and the many low‑variance lexical modes need little refinement, target Gen‑PPL is reached in **fewer flow steps** than ELF/LangFlow at equal size — few‑step generation should *improve*, since coarse‑first is what few‑step sampling rewards.
- **C5 — it wins where meaning‑order matters.** Largest gains on long‑form coherence, outline‑faithful and length‑controlled generation, and infilling/planning — the regime the essay predicts, and where surface‑first generation hurts most.

**Ablations.** spectrally‑shaped space vs raw ELF space (the core claim — same flow, only coordinates change); remove $\mathcal L_{\text{order}}$ (split but unordered → does the band edge blur and the schedule lose monotonicity?); remove $\mathcal L_{\text{within}\downarrow}$ (does surface variance leak back into top modes and reinstate surface‑first?); free spectral bias only vs + optional per‑mode schedule (d); source of the meaning signal (sentence encoder vs back‑translation vs entities).

## 8. Risks

- **The spectrum may not separate cleanly.** Meaning and surface variance could be entangled in a way no orthonormal re‑sorting untangles (no elbow). C1 is the early tripwire, on a frozen encoder, before any expensive training.
- **Total‑covariance is a linear lens; meaning may be nonlinear.** Mitigation: the eigenbasis is of a *learned* $\phi$ (so the geometry can be curved into linearity), and a kernelized / local‑covariance variant if needed. The law $\tau_k\propto\lambda_k^{-1}$ is cleanest linearly but the *ordering* it implies is observed in deep models too.
- **The meaning signal defines "semantic" externally** (shared with any essay‑faithful method, incl. SCF). We test sensitivity to its source (C5 ablation) and treat fully unsupervised spectral shaping as the harder follow‑up.
- **Worst case** the shaped space reduces to ELF's and we recover an ELF‑class model — **bounded downside**, and C1 tells us cheaply whether we are in that case.

---

## References

**From this folder (lessons taken and surpassed)**
- *ELF: Embedded Language Flows* — Hu, Qiu, Lu, Zhao, Li, Kim, Andreas, He (MIT). "Stay in continuous space; discretize only at the end." We keep this and leave its Flow Matching *entirely untouched* — we re‑tune only the space's spectrum. https://arxiv.org/abs/2605.10938 · code: https://github.com/lillian039/ELF
- *LangFlow: Continuous Diffusion Rivals Discrete* — learns a single **scalar** noise schedule (info‑uniform, Gumbel). PRISM generalizes "schedule" to a **spectrum** induced by geometry. https://arxiv.org/abs/2604.11748
- *Continuous Latent Diffusion Language Model (Cola)* — a **compressed** semantic latent via Text‑VAE + block‑causal prior; separates semantics by *bottleneck*, not by ordering the diffusion's mode‑emergence spectrum. Positioned against in §6. https://arxiv.org/abs/2605.06548
- *Semantic Corruption Flows* (this folder's proposal) — answers the essay by **learning the forward corruption**; PRISM answers by **leaving physics alone and moving the coordinates**. The complementary, and we argue simpler, lever.

**The mechanism this proposal rests on**
- *Spectral Bias in the Learning/Sampling Dynamics of Diffusion Models* — the exact law $\tau_k\propto\lambda_k^{-1}$: high‑variance modes emerge first. The fact that turns "the schedule is missing" into "the schedule is misaligned." https://arxiv.org/abs/2503.03206
- Classical backbone: total‑covariance / between‑within decomposition (Fisher LDA); nested‑subspace ordered latents (nested dropout) — repurposed here from *discrimination* to *generation order*.

**Conceptual precedent (non‑Gaussian / structured order — images)**
- *Blurring Diffusion* / *Inverse Heat Dissipation* (Hoogeboom & Salimans, 2022; Rissanen et al., 2022) — coarse‑to‑fine via a *fixed spatial‑frequency* spectrum. PRISM is the language analog with a **learned semantic** spectrum, and it does not require changing the forward process to get the ordering.

**The essay (provided as screenshots)**
- *noise schedule ≠ generation schedule*; *Generation space defines what a state is / Generation schedule defines how states should move*; the *stop‑halfway* diagnostic; $u=u_{\text{sem}}+u_{\text{lex}}$ with $u_{\text{sem}}$ to be resolved before $u_{\text{lex}}$; `meaning → structure → text`; *"learn **where** meaning becomes text."*

---

*Image diffusion never needed a clever schedule — it got coarse‑first for free because pixel variance is coarse structure. Language got the opposite spectrum, so the same physics generates surface‑first: the essay's `noise → tokens`. PRISM's claim is that the generation schedule was never the missing piece — the **coordinates** were. Re‑tune the latent so its variance spectrum is the semantic spectrum, and `meaning → structure → text` is what diffusion does on its own, with "where meaning becomes text" reported as the band edge $K^\star$. C1 decides it on a frozen encoder, before a single flow step is trained.*
