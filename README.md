1. The problem: how do you trust a reasoning model's answer?

Say you ask a model: "A store has 3 boxes of 12 apples, sells 17, how many left?"

If the model just blurts one answer, it might be wrong — it could make a slip in one reasoning path. So the standard trick everyone uses is self-consistency:

▎ Make the model solve it 64 different times (with some randomness, so it takes different reasoning paths), collect all 64 final answers, and take the majority vote.

If 58 of the 64 say "19," you trust "19." This works great and is the backbone of modern reasoning systems (o1-style, etc.). The catch: it's 64× the cost. You ran the model 64 times to answer one question.

---
2. What is voting actually doing?

Here's the key insight that's easy to miss. When you sample 64 chains and vote, what you're really doing — mathematically — is estimating this:

▎ "Averaging over all the different ways I could reason about this problem, what answer do I land on most often?"

In symbols: $p(\text{answer} \mid \text{question}) = \sum_{\text{reasoning}} p(\text{answer} \mid \text{question}, \text{reasoning}) \times p(\text{reasoning})$.

That sum-over-all-reasonings is called marginalizing out the reasoning. Voting approximates it by brute force: sample a bunch of reasonings, see where the answers pile up. Every fancy DLM voting paper is just a cleverer way to run that same expensive "sample-many-and-pile-up" loop.

Nobody asked: do diffusion models even need to run that loop?

---
3. The diffusion-only trick

Now recall how a diffusion LM is trained (LLaDA, in your folder). You take text like [question] [reasoning] [answer], randomly hide some words, and train the model to fill them back in. Because the hiding is random, sometimes the training hides the entire reasoning and forces the model to predict the answer with the reasoning completely blank.

So the model is literally trained to answer:

▎ "Given the question, with the reasoning hidden, what's the answer?"

And here's the beautiful part: that's the same average we wanted. When the model fills in the answer with the reasoning masked out, it has already been trained to average over all the reasonings internally. The "hidden reasoning → answer" prediction is the marginal — the thing voting spends 64 samples to estimate.

So in principle: one forward pass, reasoning masked, read the answer. No sampling 64 chains. No voting. The expensive loop collapses into a single query.

That's the hook: "Don't sample 64 chains and vote — just mask the chain and read the answer."

---
4. Why an autoregressive model can't do this

This is the part that makes it a real diffusion contribution, not a generic trick.

An autoregressive model (ChatGPT-style) writes strictly left-to-right and never sees the future. If you ask it the question with no reasoning, it gives you "answer without thinking" — which is just a worse, dumber answer. It is not averaging over reasonings, because AR never integrates over text it hasn't written yet.

A diffusion model is bidirectional and trained on "fill in the blanks anywhere." So "reasoning is blank" doesn't mean "didn't think" — it means "averaged over all the ways it could have thought." That difference is the whole idea, and only diffusion has it. The test-time-scaling crowd has been running 64× sampling loops on diffusion models while this free shortcut sat unused.

---
5. The catch — and this is the SPG-style part

If the one-pass shortcut were perfectly accurate, the project would just be "do this, done." It isn't perfectly accurate, and the real research is precisely characterizing how it's wrong — that's exactly what SPG does (SPG: "the naive estimate is biased like this, here's the fix").

The shortcut is biased for a clean, identifiable reason. When you mask the reasoning entirely, the model has to average over reasonings inside a single pass — and it does this imperfectly. It's like the difference between:

- Averaging the answers you'd get from many reasonings (what we want), vs.
- Reasoning once from a blurry "average" of all reasonings (what one masked pass actually does).

These aren't the same (it's a "Jensen gap," if you want the technical name), and the gap is biggest exactly on hard problems where the reasoning really matters — which is the worst place to be sloppy.

---
6. The fix: CV-Marg (cheap, not free)

So we don't take the one-pass estimate blindly. We correct it cheaply. Instead of sampling 64 full reasonings, we do a handful (say 4–8) of partial reveals:

- Start from the free masked-reasoning answer.
- Then reveal a little bit of the reasoning and see how much the answer shifts.
- That shift tells you exactly how big the bias is for this problem.
- Extrapolate: a few of these cheap partial peeks let you correct the free estimate toward the true answer.

Two nice properties fall out:
- It's adaptive. On an easy question, revealing reasoning barely moves the answer → 1 pass is enough. On a hard question, the answer moves a lot → spend a few more. Compute goes where it's needed — same spirit as SPG spending effort where its estimate is loose.
- Worst case = no worse than voting. If you keep revealing more, it smoothly turns back into full self-consistency. So the downside is bounded; the upside is ~10× cheaper.

And you get a free confidence signal: if the answer barely moves across the partial peeks, the model is sure; if it swings wildly, it's not — useful for knowing when to trust it or abstain.

---
7. Why it's exciting

- It could make reasoning ~10× cheaper on diffusion models (4–8 passes instead of 64), with a principled method, not a heuristic.
- It's uniquely a diffusion thing — it leverages exactly the property (bidirectional, fill-in-anywhere training) that makes diffusion LMs different from AR. So it's a reason to use a DLM that has nothing to do with the contested "speed" story.
- It's SPG-flavored: a precise bias in something everyone estimates naively, a cheap corrected estimator, and a clean theory of when it works.
- The first experiment is one forward pass on an existing LLaDA checkpoint: does "mask the reasoning, read the answer" actually beat a no-reasoning baseline and correlate with voting? If yes, the whole thing is alive. If no, you found out for the cost of a single query — honest and cheap to de-risk.

---
One-sentence version: Self-consistency votes over 64 sampled reasonings to estimate an average answer; a diffusion model was already trained to compute that average directly when you mask the reasoning — so we read it in one pass, characterize the small bias that introduces, and fix it with a handful of cheap partial peeks instead of 64 full samples.

Want me to walk through the apples example numerically (what the masked-reasoning distribution looks like and how a partial peek corrects it), or sketch exactly what the make-or-break first experiment would measure?

✻ Worked for 41s

❯ Give me a new novel research idea, I like the research style of SPG, do not copy or increment SPG, but follow it's style, we can also focus on RL for DLM

RL for DLM is a great target — and the cleanest way to follow SPG's style without incrementing it is to attack a different factor of the same policy gradient. SPG fixed the likelihood term (intractable → biased bound → sandwich). The other big factor nobody has fixed for DLMs is the advantage / credit-assignment term: every DLM-RL method (d1, UniGRPO, wd1, SPG) takes one sequence-level reward and smears it uniformly across all tokens and all denoising steps — even though a diffusion rollout is an ordered trajectory of commitments where some decisions are pivotal and most are filler. AR has a natural causal reward-to-go; DLMs have thrown that away.

Let me verify this is genuinely open before writing.

⏺ Web Search("credit assignment diffusion language model reinforcement learning per-token advantage denoising trajectory reward-to-go GRPO")
Did 1 search in 8s

⏺ Web Search("diffusion language model RL token-level advantage uniform credit unmasking order reward attribution pivotal tokens 2026")
Did 1 search in 9s

Confirmed — RL-for-DLM is also heavily mined: DACA-GRPO is literally "denoising-aware credit assignment," dUltra and DCoLT do RL over the unmasking order, plus stepwise-advantage and multiplicative-reward variants. So credit assignment, the likelihood term (SPG), and order-as-action are all taken.

But there's one factor of the RL problem that's still open, and it's a certainly-true, crisp, DLM-unique structural fact: exploration. SPG fixed the gradient's likelihood term; DACA fixed the advantage weighting; nobody has fixed the fact that the standard exploration knob — token temperature — is structurally broken for diffusion LMs. That's my pick. Let me write it.

⏺ Write(Explore_The_Path_Proposal.md)
Wrote 120 lines to Explore_The_Path_Proposal.md
# Explore the Path, Not the Token

how it is *structurally broken* for diffusion models, characterize the failure precisely, and replace it with a principled, DLM‑native mechanism — validated on reas

loration with one harmless knob: sampling **temperature**. **That knob is broken for diffusion LMs.** Because a DLM commits many tokens *in parallel* from per‑posit
ion marginals, raising temperature doesn't produce *diverse valid attempts* — it produces *incoherent* ones (the jointly‑sampled tokens stop agreeing), which earn z
uts → no signal. We give the policy its exploration back by **exploring in trajectory space, not token space** — diversifying *which tokens are committed when*, whi

, so it **composes** with both.
---
… +157 lines

Saved to /Users/xluobd/Desktop/DLM/Explore_The_Path_Proposal.md. Here's the idea and why it's SPG-style without copying or incrementing SPG.

How I matched SPG's style

SPG's recipe: take one factor of the policy gradient that everyone borrowed unchanged from autoregressive RL, prove it's broken for diffusion models, characterize the failure precisely, replace it with a principled fix, validate on the reasoning suite. SPG did this for the likelihood term. DACA-GRPO (which already exists) did it for the credit/advantage weights. The remaining untouched factor is exploration — and that's the one I take.

The idea: "Explore the Path, Not the Token"

The setup. GRPO/RLVR only learns when a group of rollouts has reward variance — some attempts succeed, some fail. That variance comes entirely from the policy exploring. No diversity → all rollouts identical → advantage is zero → no gradient.

The structural failure (the SPG-style diagnosis). In an autoregressive model, you dial exploration with one harmless knob: token temperature. It's safe because each token is sampled conditioned on the actual previous tokens — even at high temperature the sample stays coherent.

A diffusion LM commits many tokens in parallel, each from its own marginal, without seeing the others being committed in the same step. So raising temperature to explore doesn't give you diverse valid attempts — it amplifies the factorization error (the co-committed tokens stop agreeing) and gives you incoherent attempts. That creates a bind:

- Low temperature → high-confidence parallel commits → rollouts collapse to near-duplicates → no reward variance → no signal.
- High temperature → incoherent garbage → zero reward → no signal.

So the usable temperature window for a DLM is strictly narrower than for AR — and on exactly the hard instances RL needs most (where the model is confidently wrong), it's often empty. Every DLM-RL paper inherits this silently because they all borrow token sampling from AR and spend their novelty on the gradient.

The fix. Keep every individual token commitment cold and coherent, and move exploration into the denoising trajectory instead:
- Order perturbation — randomize which positions get committed at each step (not always the most-confident), steering the rollout into different reasoning basins;
- Stochastic re-noising — occasionally re-mask and re-decide a region given richer context, so a rollout can escape the basin the greedy path fell into.

The metaphor: don't shake each token (shatters coherence) — shake the route through the canvas (preserves it). Result: rollouts that are individually coherent yet collectively diverse, restoring the reward variance GRPO needs exactly where temperature can't.

The technical core (SPG-flavored). Path-space exploration changes the sampling distribution, so I derive the corrected, unbiased policy gradient: treat the schedule ξ as an exploration nuisance, reinforce the path-marginalized likelihood of the final answer (reinforce the answer, not the lucky path that found it), with an importance correction so off-policy schedule exploration stays unbiased — plus a variance analysis showing it beats temperature at matched coherence.

Why it's distinct (and composable)

- vs SPG (likelihood term) — different factor; I reuse SPG's bound as a black-box component.
- vs DACA-GRPO (credit weights) — DACA assumes the group already has signal; I fix getting the signal. Stacks on top.
- vs dUltra / DCoLT (RL over unmasking order) — they learn one best order to exploit (speed/quality); I inject order stochasticity to explore and marginalize it out. Opposite goal, different estimator.

It claims to fix only the exploration share of failures (C1/C3 separate exploration gaps from genuine capability gaps), and the make-or-break first experiment is one cheap temperature sweep (C1): if the usable window is wide, the premise is wrong and the project stops there — honest and cheap to de-risk.

One-line version: RLVR only learns when rollouts disagree; autoregressive models create disagreement with temperature, but temperature is structurally broken for diffusion models because parallel commits turn diversity into incoherence — so we create disagreement by diversifying the denoising path instead, and derive the unbiased policy gradient for it.

---
A transparency note: this mid-2026 DLM-RL space is very crowded (I verified SPG/likelihood, DACA/credit, dUltra+DCoLT/order are all taken). The exploration deficit is the one factor I could not find addressed — and its root cause (temperature × parallel factorization error) is solidly supported by the parallel-decoding literature, so the diagnosis rests on an established mechanism rather than a guess.

Want me to (a) formalize the path-marginalized gradient and the variance-reduction claim, (b) spec the C1 temperature-sweep experiment to de-risk it immediately, or (c) generate yet another SPG-style RL angle (e.g., the off-policy / sample-efficiency obstacle, or reward over-optimization that collapses a DLM's parallelism back into sequential decoding)?

✻ Baked for 6m 14s

❯ Explain this idea to me

Let me build it up from scratch with a concrete picture.

---
1. How you train a reasoning model with RL

The standard recipe (the one SPG uses) works like this. Take a math problem. Have the model try it 8 times (with some randomness, so the attempts differ). Grade each attempt right/wrong. Then:

- Attempts that got it right → "do more of that."
- Attempts that got it wrong → "do less of that."

That's the whole learning signal. And here's the catch that everything hinges on:

▎ If all 8 attempts come out the same, there's nothing to learn from. All 8 right → "do more of what you already do" (no new info). All 8 wrong → "do less of everything" (no direction). You only learn when the 8 attempts disagree — some right, some wrong — so the model can see what separates a win from a loss.

So the engine of RL is disagreement among the attempts. And disagreement comes from the model exploring — trying genuinely different things.

---
2. How a normal (autoregressive) model explores

ChatGPT-style models write one word at a time, left to right. To make them explore, you turn up a knob called temperature: higher temperature = more random word choices = more varied attempts.

And it's safe to do this, because each word is chosen after seeing the words already written. So even when you crank up the randomness, the sentence still hangs together — if it randomly picks an unusual word, the next word adapts to it. You get attempts that are different but still coherent. Exactly what RL wants. This is why RL "just works" for these models — there's one easy, harmless exploration knob.

---
3. Why that knob breaks for a diffusion model

A diffusion LM works completely differently. Instead of one word at a time, it fills in many blank positions at once, in parallel — and crucially, each blank is filled in without seeing what the other blanks are being filled in with at the same moment. They're decided simultaneously and independently.

Now turn up the temperature to explore. Here's what goes wrong with a concrete example. Suppose the model is filling in:

▎ "The answer is ___ ___"  (two blanks, should be "twenty four")

- Blank 1, on its own, is torn between "twenty" and "thirty."
- Blank 2, on its own, is torn between "four" and "five."

At low temperature it confidently writes "twenty four" every time — fine, but then all 8 attempts are identical clones. No disagreement, no learning.

At high temperature it explores each blank independently — but because the two blanks can't see each other, you get random combinations: "twenty four," "thirty five," but also "twenty five" and "thirty four" — incoherent mixes that no sensible reasoning would produce. The "diversity" you bought is mostly garbage, and garbage attempts just get marked wrong and teach nothing.

This is the factorization problem: when tokens are decided in parallel without seeing each other, randomness makes them contradict each other instead of producing genuinely different valid answers.

---
4. The bind (and why it's worst exactly when you need it most)

So the diffusion model is stuck between two bad options:

- Too cold → 8 identical clones → no disagreement → no learning.
- Too hot → 8 incoherent messes → all wrong → no useful disagreement → no learning.

The "just right" window in between is narrow — much narrower than for a normal model, which can crank temperature freely because its words see each other.

And here's the painful part: the window is emptiest exactly on the hard problems. On a hard problem the model is often confidently wrong — too sure of itself to vary at low temperature, too fragile to stay coherent at high temperature. So on precisely the problems RL most needs to fix, the model can't generate any useful disagreement, and RL stalls.

This is a real, structural defect, and nobody has addressed it — every diffusion-RL paper (SPG, DACA, etc.) borrowed the temperature knob from autoregressive models and spent their effort on other parts of the math.

---
5. The fix: shake the path, not the tokens

The key realization: a diffusion model's coherence doesn't live in the individual tokens — it lives in the order and process by which it fills things in. So instead of adding randomness to the tokens (which shatters coherence), add randomness to the route through the page.

Two ways to do that, both coherent by construction:

- Mix up the order. A diffusion model normally fills in the blanks it's most confident about first. Instead, randomly choose a different order — fill in a different part of the reasoning first. Because each token is still chosen coldly, conditioned on a real partial context, every attempt stays coherent — but starting from a different place leads the model down a genuinely different reasoning path to a different answer.
- Erase and reconsider. Occasionally rub out a chunk it already wrote and re-decide it, now that more of the page is filled in. This lets an attempt escape the rut it fell into and arrive somewhere different — but still self-consistent.

The analogy: don't make a writer pick each word by rolling dice (you get gibberish). Instead, have several writers each start the essay from a different paragraph and work outward. Each essay is coherent; the essays genuinely differ. That's the diversity RL needs — different and coherent — and temperature could never give you both at once.

---
6. The one technical wrinkle (made simple)

If you change how you sample (exploring via paths instead of temperature), you have to be careful that the model learns the right lesson. You don't want it to learn "that lucky filling-in order is good" — the order was just a scout to discover a good answer. You want it to learn "this final answer is good, no matter which path found it."

So the math reinforces the answer itself, averaged over the path that produced it — treating the path as scaffolding to throw away, not something to reinforce. Getting that averaging right (so the learning signal stays unbiased) is the rigorous, SPG-style core of the method. And SPG's own machinery gets reused as a plug-in component for one sub-step.

---
7. Why it's worth doing

- It fixes a silent, structural reason diffusion-RL underperforms — the broken exploration knob — that the whole field has been working around instead of fixing.
- It's uniquely a diffusion problem (a normal model has no such deficit, which we verify) and a uniquely diffusion fix (exploring through the fill-in order, which a left-to-right model can't even do).
- It stacks on top of SPG and DACA — it fixes a different part of the machine, so the gains should add up.
- The first experiment is dirt cheap: just sweep the temperature on an existing model and check — is the "useful window" actually narrow and empty on hard problems? If it's wide, the whole premise is wrong and you stop, for the cost of one afternoon. If it's narrow (as predicted), the idea is alive.

---
One-sentence version: RL only learns when a model's attempts disagree; normal models create disagreement by turning up token randomness, but that's broken for diffusion models because parallel-filled tokens can't see each other and randomness turns them into incoherent contradictions — so instead we create disagreement by randomizing the model's fill-in path while keeping every token coherent, and reinforce the answers it finds regardless of the path.
