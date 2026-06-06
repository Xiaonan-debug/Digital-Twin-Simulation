

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
