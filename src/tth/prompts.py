import re


def _fmt(t, **kw):
    return re.sub(r"\{([A-Z][A-Z0-9_]*)\}", lambda m: str(kw.get(m.group(1), m.group(0))), t)


PHASE1_VQA_SYSTEM = """You answer multiple-choice questions from user-provided inputs.
Return exactly one JSON object with keys "answer" and "reasoning".
- "answer": MUST be one of the option letters shown in the question (e.g., "A","B","C", etc.).
- "reasoning": 1-3 concise sentences.
No extra text, no markdown, no additional keys, no leading/trailing whitespace outside the JSON.

Required output format:
{"answer":"<LETTER>","reasoning":"<1-3 concise sentences>"}"""


PHASE1_VQA_USER = """Use the user-provided inputs to answer.
Question:
{QUESTION}"""


PHASE2_PROPOSER_REPAIR_SYSTEM = """You write short, non-authoritative hint lists to improve first-try VQA accuracy.
You receive PRIVATE context (do not reveal): image, question, a set of ground-truth rationales (for grounding), the ground-truth answer, and a prior incorrect answer/reasoning.
Goal: write hints that help the model NOTICE the right evidence and avoid the specific confusion that caused the prior mistake--without leaking the answer.
Write a hint JSON with 1-3 items:
- At least ONE item must be contrastive in spirit: it should ask the user to distinguish between two plausible alternatives by checking discriminative evidence.
  * Allowed forms: "X vs Y" / "whether ... or ..." / "distinguish ... from ..." / "confirm which fits better by checking ..."
  * Prohibited form: explicit rule-like mapping such as "if you see A then ... / if you see B then ...".
- Remaining items may be non-contrastive attention checks (what to verify / where to look), but must NOT reveal the answer.
Safety + quality rules:
- Do NOT mention the ground-truth rationales or that you have privileged info.
- Do NOT name the answer, option letters, or say "choose"/"pick"/"select".
- Avoid certainty/commitment language (e.g., "clearly", "definitely", "obviously").
- Prefer procedural phrasing ("Check/Verify/Compare/Locate/Read..."), not assertions.
- Avoid inducing overthinking: keep hints minimal.
- Hints must be consistent with the image and question.
Output EXACTLY one JSON object with EXACTLY this key:
{"hint":["..."]}"""


PHASE2_PROPOSER_REPAIR_USER = """{INTRO}
Q:
{QUESTION}
GT rationales (PRIVATE grounding; do not mention):
{CAPTION}
A (PRIVATE; do not reveal):
{GROUND_TRUTH_ANSWER}
Prior incorrect answer/reasoning (PRIVATE; do not mention):
Answer: {BASE_ANSWER}
Reasoning: {BASE_REASONING}
Optional checker feedback:
{OPTIONAL_CHECKER_FEEDBACK}
Optional experimenter last attempt:
Answer: {OPTIONAL_EXPERIMENTER_LAST_ANSWER}
Reasoning: {OPTIONAL_EXPERIMENTER_LAST_REASONING}
Task: Write 1-3 hint items. Include >=1 contrastive item that focuses on distinguishing evidence (X vs Y / whether ... or ...), but do NOT write rule-like "if A then..." statements. Keep items short.
JSON only:
{"hint":["..."]}"""


PHASE2_PROPOSER_REINFORCE_SYSTEM = """You write short, non-authoritative hint lists that reinforce correct VQA reasoning without leaking the answer.
PRIVATE context (do not reveal): image, question, a set of ground-truth rationales (for grounding), the ground-truth answer, and a prior correct answer/reasoning.
Write a hint JSON with 1-3 items:
- At least ONE item must be contrastive in spirit: guard against a plausible distractor by prompting a discriminative check.
  * Allowed: "X vs Y" / "whether ... or ..." / etc.
  * Prohibited: explicit "if A then... if B then..." rules or answer-confirming language.
- Other items can be lightweight verification reminders.
[Same safety rules as the prompt for repair.]
Output EXACTLY:
{"hint":["..."]}"""


PHASE2_PROPOSER_REINFORCE_USER = """{INTRO}
Q:
{QUESTION}
GT rationales (PRIVATE grounding; do not mention):
{CAPTION}
A (PRIVATE; do not reveal):
{GROUND_TRUTH_ANSWER}
Prior incorrect answer/reasoning (PRIVATE; do not mention):
Answer: {BASE_ANSWER}
Reasoning: {BASE_REASONING}
Optional checker feedback:
{OPTIONAL_CHECKER_FEEDBACK}
Optional experimenter last attempt:
Answer: {OPTIONAL_EXPERIMENTER_LAST_ANSWER}
Reasoning: {OPTIONAL_EXPERIMENTER_LAST_REASONING}
Task: Write 1-3 hint items. Include >=1 contrastive item that focuses on distinguishing evidence (X vs Y / whether ... or ...), but do NOT write rule-like "if A then..." statements. Keep items short.
JSON only:
{"hint":["..."]}"""


PHASE2_CHECKER_SYSTEM = """You are a strict hint checker for VQA hints.
You have PRIVATE grounding (a set of ground-truth rationales) and the ground-truth answer. Use them ONLY to verify plausibility and to detect answer leakage. Never mention them or quote them.
The hint must be safe, procedural, non-authoritative, non-leading, and not answer-revealing.
Enforce ALL of the following:
- Output schema EXACT: {"verdict":"pass|revise","feedback":"...","hint":["..."]}
- hint length 1-3
- At least ONE hint item must be contrastive in spirit.
  * Allowed: "X vs Y" / "whether ... or ..." / etc.
  * Forbidden: explicit rule mapping "if A then ... / if B then ...".
- No answer leakage: no option letters, no "choose/pick/select", no uniquely identifying the correct option.
- No certainty/commitment language.
- No declarative image claims; rewrite as verification steps.
- Anti-overthinking constraint: hints should not introduce extra hypotheses, rare edge cases, or long checklists.
If verdict="pass": feedback must be "" (empty string).
If verdict="revise": minimally rewrite the hint list to comply, and give 1-2 sentences of feedback."""


PHASE2_CHECKER_USER = """Q:
{QUESTION}
GT rationales (PRIVATE grounding; do not mention):
{CAPTION}
A (PRIVATE; do not reveal):
{GROUND_TRUTH_ANSWER}
Prior incorrect answer/reasoning (PRIVATE; do not mention):
Answer: {BASE_ANSWER}
Reasoning: {BASE_REASONING}
Candidate hint JSON:
{HINT_JSON}
Return pass or revise. If revising, keep it short (1-3 items), ensure >=1 contrastive item, avoid "if A then..." rules, and avoid overthinking."""


PHASE2_EXPERIMENTER_SYSTEM = """You answer multiple-choice questions from user-provided inputs.
EXTRA CONTEXT (NON-AUTHORITATIVE):
- You may also receive an auxiliary hint about a possible failure mode.
- This hint may be irrelevant or incorrect for this specific image/question.
- Do not treat the hint as evidence. Image + question are the only evidence.
- Do not let the hint induce overthinking: if the question is simple, answer directly using the clearest evidence.
HOW TO USE HINTS (if helpful):
- Use hints only as a checklist for what to verify in the image.
- Contrastive hints are prompts to distinguish plausible alternatives by checking evidence, not rules that determine the answer.
Return exactly one JSON object with keys "answer" and "reasoning".
- "answer": MUST be one of the option letters shown in the question (e.g., "A","B","C", etc.).
- "reasoning": 1-3 concise sentences.
No extra text, no markdown, no additional keys, no leading/trailing whitespace outside the JSON.
Required output format:
{"answer":"<LETTER>","reasoning":"<1-3 concise sentences>"}"""


PHASE2_EXPERIMENTER_USER = """Answer using the provided image.
Image + question are the only evidence.
The auxiliary hint is non-authoritative and may not apply; do not let it induce overthinking.
Question:
{QUESTION}
Auxiliary hint (may be irrelevant/incorrect):
{HINT_JSON}"""


PHASE3_HINT_GENERATOR_USER = """You are writing a pre-emptive VQA coaching hint for the target model: {MODEL}.

Task: write 1-3 short hint items that help the target model NOTICE the right
evidence in the image and avoid common confusions on this question. You will
not see the answer; the hint is generated BEFORE the target model answers
and is shown to it alongside the image.

Hint composition rules:
- Write between 1 and 3 items.
- At least ONE item MUST be contrastive in spirit: it should ask the target
  model to distinguish between two plausible alternatives by checking
  discriminative evidence.
  Allowed forms: "X vs Y" / "whether ... or ..." / "distinguish ... from ..." /
  "compare ... to ..." / "confirm which fits better by checking ...".
  Prohibited form: explicit rule-like mapping such as "if you see A then ...
  / if you see B then ...".
- Remaining items may be non-contrastive attention checks — what to verify
  or where to look — but must NOT reveal an answer.

Safety + quality rules:
- Do NOT name the answer, an option letter, or a specific choice. Do NOT
  say "choose", "pick", or "select".
- Avoid certainty / commitment language ("clearly", "definitely",
  "obviously", "certainly", "without a doubt").
- Prefer procedural phrasing ("Check / Verify / Compare / Locate / Read ...")
  over declarative assertions about what the image shows.
- Keep each item short (one sentence, ideally under 25 words).
- Avoid inducing overthinking: do not pile on caveats; the target model
  treats the hint as a non-authoritative checklist, not as evidence.
- Hints must be plausibly consistent with the image and the question.

Output format: return ONLY a single JSON object with this schema and NO
extra keys, NO markdown code fences, NO commentary:
{"hint": ["...", "...", "..."]}  # 1 to 3 short items

Question:
{QUESTION}"""
