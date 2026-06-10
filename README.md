# On-Policy Distillation of LFM2 on the Sharp CV-P09FX AC Manual

A from-scratch, heavily-commented implementation of **on-policy distillation** (the
context-distillation variant), built for practice on a free Colab T4. It reuses the
Sharp CV-P09FX air-conditioner Q&A task from a prior QLoRA SFT experiment:
<https://github.com/kakopappa/Fine-tuning-LFM2.5-350M-on-Sharp-CV-P09FX-AC-Manual>

## What this does

A small **student** (`LiquidAI/LFM2.5-350M`) knows nothing about the manual. A larger frozen
**teacher** (`LiquidAI/LFM2-1.2B`) also doesn't — but if you paste the manual into the teacher's
*prompt context*, it answers well. On-policy distillation moves that ability into the student's
weights **without the student ever seeing the manual**:

1. **Student** samples its own answer to a question — *no manual in context* (on-policy rollout).
2. **Teacher** scores those exact tokens — *with the manual in context* (frozen, no grad).
3. Loss = per-token **reverse KL** `D_KL(student ‖ teacher)` over the student's own tokens.
4. Backprop into the student (LoRA only). The knowledge ends up in the student's weights.

Both models are LFM2-family, so they share one tokenizer (`vocab_size = 65536`). That shared vocab
is what makes token-level KL valid — the script asserts tokenizer alignment at startup.

```
                  question
                     │
        ┌────────────┴─────────────┐
        ▼                          ▼
  STUDENT prompt              TEACHER prompt
  (no manual)                 (manual in context)
        │ sample                    │
        ▼                          │
   gen tokens  ───────────────────►│  score the SAME tokens
        │                          │
        ▼                          ▼
   student logits  ◄── reverse KL ──►  teacher logits   (frozen)
        │
        ▼
   backprop into student LoRA
```


## How to run (Colab)

1. **Runtime → Change runtime type → T4 GPU.**
2. Run the install cell (it also removes Colab's preinstalled `torchao`, which clashes with `peft`).
   **Restart the session**, then run from the top.
3. Run cells in order: imports/config → data loaders → tokenizer+models → prompt builders →
   distill step → eval helpers → load data → **Before** → **Train** → **After**.
4. The trained student LoRA adapter is saved to `./lfm25_350m_ac_onpolicy_lora`.

### Key configuration knobs

| Knob | Default | Notes |
|------|---------|-------|
| `STUDENT_ID` / `TEACHER_ID` | LFM2.5-350M / LFM2-1.2B | Must share a tokenizer. |
| `MAX_NEW_TOKENS` | 128 | Rollout length. Lower (≈64) concentrates the signal on fact tokens. |
| `EPOCHS` | 3 | Tiny dataset (~45 questions) → bump to 20–25 to actually memorize facts. |
| `ACCUM_STEPS` | 8 | Gradient accumulation (effective batch). Lower → more update steps. |
| `LR` | 1e-4 | 2e-4 trains faster on this small set. |
| LoRA `r` | 16 | Bump to 32–64 for more fact-storage capacity. |
| `MAX_MANUAL_TOK` | — | Keep the teacher context **short** (see below). |

## The teacher context matters most

The single most important thing is that the **teacher must actually answer correctly** before you
train — the student can only become as good as the teacher (*garbage in, garbage out*). Two traps:

- **Don't feed the raw bilingual PDF.** The French boilerplate pushes the real specs past the token
  cap, and a long context (8000+ tokens) **OOMs the T4** because eager attention memory grows with
  the square of sequence length. Use `attn_implementation="sdpa"` *and* keep the context short.
- Build a **compact, fact-only English context** (`build_relevant_context`): keep only manual lines
  that mention measurements/parts (`inch`, `mm`, `width`, `screw`, `559`, …). ~2.5k tokens, fast,
  no OOM, and it contains the specs.

Always verify with a probe before training:

```python
q = "What is the minimum window width for the window panel?"
pt = build_prompt_ids(q, with_manual=True, manual_text=manual)
out = teacher.generate(pt, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(out[0, pt.shape[1]:], skip_special_tokens=True))
# expect ~"22 inches (559mm)" — not "1.5 inches"
```

## Evaluation

- **`reverseKL` (training log):** mean per-token `D_KL(student ‖ teacher)` on the student's sampled
  rollouts, in nats/token. `0` = identical to teacher. Noisy (fresh samples each step) — read the
  trend, not single points.
- **Held-out reverse-KL (`eval_kl`):** the same quantity but **greedy** decoding on **held-out** val
  questions → deterministic, so Before vs After is a fair comparison. Lower = closer to the teacher.
- **Qualitative answers:** print a few greedy student answers before/after — the real test.

To save the best checkpoint, early-stop on the held-out `eval_kl` (per epoch), not on the noisy
training `reverseKL`.

## Results & lessons learned

On a clean run (fresh student, correct teacher, ~85 steps), reverse-KL fell **0.99 → ~0.50**, and
*style/behavior* transferred well (fluency, structure, fragments like "2 screws" appearing in the
student's answers). But the **specific facts did not fully land** (e.g. window width stuck near
"1 inch" instead of 22"/559mm). Why:

- A rollout is ~100 filler tokens around **one** fact token. The per-token-averaged KL is dominated
  by the easy filler tokens that already match, so the rare fact token's gradient gets diluted.
- Memorizing a fixed fact set into 350M weights via a low-rank LoRA needs many more passes.

**Takeaway:** on-policy distillation excels at transferring **skills / behavior / reasoning** and
teaching a model to recover from its own mistakes. For injecting a fixed set of rare **facts**, it's
an inefficient channel — direct **SFT on the answer tokens** (the original repo's approach) is the
better tool. To push facts in anyway: `r=32–64`, `EPOCHS≈25`, `MAX_NEW_TOKENS≈64`.

## Gotchas hit (and fixed)

- **`torchao` / `peft` clash:** Colab ships `torchao 0.10`; a fresh `peft` wants `>0.16`. Fix:
  `!pip uninstall -y torchao` (unused here) + restart.
- **`apply_chat_template` returns a `BatchEncoding`** (dict) in recent transformers, not a tensor —
  `.shape` raises `KeyError: 'shape'`. Use `return_dict=True` and take `["input_ids"]`.
- **CUDA OOM on long teacher context:** eager attention is O(L²) in memory. Keep the context short
  and load the teacher with `attn_implementation="sdpa"`.

## Requirements

`transformers>=4.56`, `peft>=0.13`, `accelerate`, `pypdf`, `requests`, and a CUDA GPU
(free Colab T4 is sufficient). Local training needs a GPU larger than ~4 GB.
