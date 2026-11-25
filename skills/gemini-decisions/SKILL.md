---
name: gemini-decisions
description: "Use when classifying, routing, or triaging messages/files. Calls gemini-decision-1."
version: 0.1.0
author: Matthijs van den Berg
license: MIT
---

# Gemini Decisions (`gemini-decision-1`)

Use `gemini_decide` and `gemini_decide_batch` whenever you need fast, calibrated closed-set judgments over text, PDFs, images, audio, or video without paying for generated text tokens.

## When to use

- **Two-stage triage:** Score a batch of 10–50 emails, chat messages, tickets, or notifications with `gemini_decide_batch` using `filter_question` and `min_probability` / `min_score`, then inspect only the high-probability hits with full reasoning.
- **Routing & intent classification:** Pick one of 2–255 fixed categories (`choice`) and branch on `confidence`.
- **Policy & guardrail checks:** Run multiple independent yes/no (`noul`) checks in one forward pass — shared input tokens are billed once regardless of question count (up to 1,000 questions).
- **Multimodal document/image qualification:** Check PDF invoices, school letters, receipts, or photos directly via `media: [{"file_path": "..."}]` (use `resolution: "low"` for coarse image triage to save tokens).

## Do NOT use for

- Generating summaries, explanations, free-text answers, or JSON field extraction (`gemini-decision-1` has an output token limit of 0).

## Question design rules

1. **Put the full question in `instructions`.** The model never sees question keys (`is_urgent`, `route`), only `instructions` and `criteria`.
2. **Respect `0.01–0.99` bounds on `noul`.** Never set a threshold above `0.99`. For high-recall triage (don't miss anything important), start around `min_probability: 0.30` and escalate to a generative model.
3. **Always include a catch-all option on `choice`** (e.g. `"other": "Anything outside the other categories"`). Choice questions distribute probability summing to 1 even when nothing fits. Avoid positional names like `none_of_the_above` because Google sorts option keys alphabetically before evaluation.
4. **Keep `score` levels linear and distinct** (2 to 10 ordered descriptions from lowest at index `0` to highest). Inspect both `score` (probability-weighted expected index) and `probabilities` when distributions might split.
5. **Ask atomic, independent questions.** Questions cannot see each other's answers; compose compound rules in code or downstream logic.
