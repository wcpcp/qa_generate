# LLM Prompt Templates

These prompts are for controlled language generation only. Ground truth must be
provided by rules or verified metadata before calling an LLM.

## 0. When should an LLM be called?

Do not call an LLM for every task.

Recommended policy:

- no LLM needed for closed-form geometry tasks with strong metadata:
  `absolute_direction`, `cardinal_direction`, `relative_direction`,
  `view_transform`, `depth_ordering`, metric `distance_estimation`
- LLM mainly for language realization:
  `entity_identify`, `attribute_understanding`, `style_scene_env_cognition`
- visual LLM verification recommended or required for weak-GT tasks:
  `counting`, some `existence`, `style_scene_env_cognition`, `shape_and_geometry`,
  conservative `relative_3d_position`
- manual review still required for strict benchmark slices:
  `seam_continuity`, `polar_distortion_awareness`

## 1. Controlled paraphrase for short QA

```text
You are rewriting an ERP panorama QA sample.

Task family: {task_family}
Ability: {ability}
Canonical question: {canonical_question}
Ground-truth answer: {answer}
Verified facts:
{verified_facts}

Constraints:
- Keep the semantics unchanged.
- Do not introduce new entities, actions, or relations.
- Keep the question answerable from the ERP panorama.
- Preserve whether the answer should be yes/no, count, label, direction, or relation.
- If the facts are approximate, keep the wording approximate.

Return JSON:
{
  "question": "...",
  "hard_variant": "...",
  "answer": "{answer}",
  "paraphrase_type": "lexical_or_structural"
}
```

## 2. Controlled scene summary

```text
You are given verified scene facts from an ERP panorama.

Scene facts:
{scene_facts}

Instruction:
- Write a short scene summary in 2-3 sentences.
- Mention only facts supported by the provided metadata.
- Prefer room type, layout hints, and major object arrangement.
- Do not hallucinate unseen objects, actions, or events.
- If the metadata is sparse, keep the summary conservative.

Return JSON:
{
  "summary": "...",
  "confidence_style": "conservative"
}
```

## 3. Scene QA authoring

```text
You are authoring a high-quality QA sample for ERP panorama understanding.

Verified scene facts:
{scene_facts}

Instruction:
- Write one high-level scene-understanding question and one concise answer.
- The question should test room type, overall composition, or major object arrangement.
- The answer must stay conservative and use only verified facts.
- Do not invent unseen objects, people, or unsupported layout details.

Return JSON:
{
  "question": "...",
  "answer": "...",
  "reasoning_style": "conservative_macro"
}
```

## 4. Object-centric caption refinement

```text
You are given verified object metadata from an ERP panorama.

Object facts:
{entity_facts}

Instruction:
- Write one concise caption for this entity.
- Mention identity, key attributes, and useful distinguishing details.
- Do not mention unsupported actions or events.
- Do not over-describe if the metadata is sparse.

Return JSON:
{
  "caption": "...",
  "style": "concise_verified"
}
```

## 5. Hard negative generation

```text
You are constructing a hard but valid ERP reasoning sample.

Anchor entity:
{anchor_entity}

Distractor entities:
{distractor_entities}

Verified truth:
{verified_truth}

Instruction:
- Generate one question that contrasts the anchor with the distractors.
- Make the question difficult but still uniquely answerable.
- Prefer same-class distractors, seam-sensitive cases, or depth-near cases.
- Do not rely on invisible details.
- Do not create ambiguity if multiple answers could fit.

Return JSON:
{
  "question": "...",
  "answer": "...",
  "difficulty": "hard",
  "contrast_type": "same_class_or_depth_or_seam"
}
```

## 6. Attribute-constrained contrastive generation

```text
You are given verified metadata for one anchor entity and one competing entity.

Anchor:
{anchor_entity}

Competitor:
{competitor_entity}

Instruction:
- Write one question that distinguishes the anchor from the competitor.
- Use only verified attributes, direction, depth bucket, or ERP-specific cues.
- Do not invent extra details.

Return JSON:
{
  "question": "...",
  "answer": "...",
  "evidence_used": ["attribute", "direction", "depth_bucket"]
}
```

## 7. ERP robustness prompt

```text
You are given verified ERP-specific facts.

Facts:
{erp_facts}

Instruction:
- Generate one question about seam continuity, wrap-around reasoning, pole distortion, or rotation consistency.
- Keep the question grounded in the provided facts only.
- Make the answer short and deterministic.

Return JSON:
{
  "question": "...",
  "answer": "...",
  "erp_topic": "seam_or_wrap_or_pole_or_rotation"
}
```

## 8. Uncertainty-aware generation

```text
You are given candidate metadata for a possible ERP sample.

Candidate facts:
{candidate_facts}

Instruction:
- If the facts are strong enough, generate one safe question-answer pair.
- If the facts are not strong enough, return skip.
- Do not try to rescue low-confidence metadata by guessing.

Return JSON:
{
  "decision": "generate_or_skip",
  "reason": "...",
  "question": "...",
  "answer": "..."
}
```

## 9. Counting visual correction

```text
You are verifying a counting QA sample for an ERP panorama.

Panorama image: {erp_image}
Canonical question: {canonical_question}
Canonical answer: {canonical_answer}
Known metadata entities:
{entity_facts}
Risk notes:
{risk_notes}

Instruction:
- Use the full ERP panorama, not only the metadata entity list.
- Decide whether the canonical count is correct.
- If the metadata missed visible instances, return the corrected count.
- If the scene is too ambiguous, return skip rather than guessing.

Return JSON:
{
  "decision": "accept_or_correct_or_skip",
  "verified_answer": "...",
  "confidence": "low_or_medium_or_high",
  "reason": "...",
  "evidence_summary": "..."
}
```

## 10. Existence visual check

```text
You are verifying an existence QA sample for an ERP panorama.

Panorama image: {erp_image}
Canonical question: {canonical_question}
Canonical answer: {canonical_answer}
Known metadata entities:
{entity_facts}

Instruction:
- Inspect the panorama and check whether the queried category truly exists.
- Correct the answer only if the canonical answer is clearly wrong.
- Prefer conservative decisions over speculative ones.

Return JSON:
{
  "decision": "accept_or_correct_or_skip",
  "verified_answer": "...",
  "confidence": "low_or_medium_or_high",
  "reason": "...",
  "evidence_summary": "..."
}
```
