# Related Work Insights Pass

Date: 2026-05-18

Scope: this pass reads the 29 locally archived papers listed in
`paper/sources/manifest.tsv`. All PDFs and text extractions were present and
non-empty at the time of the pass. The goal is not to expand the citation list;
it is to extract what should change our framing, experiments, and next research
moves. The 2026-05-18 increment adds the formal-methods ancestry and ML
accountability papers.

## Bottom Line

The strongest synthesis is:

> The literature is converging on "generate many candidates, then use a verifier,
> reward model, formal checker, or test-time adaptation loop to select or improve
> them." CRLPLRIMES is credible when framed as the small, auditable, exact-verifier
> version of that pattern: model proposes, verifier decides, trace remembers.

The multiple experiments are the point: they show that the same
verifier-bounded protocol can be instantiated across different task families,
not that ARC is the destination. The sharper claim is that exact verifiers
change the learning problem: the model should learn where to search and how to
rank certified survivors, while correctness remains outside the learned model.

The new ancestry pass sharpens the novelty boundary. The loop is not new in
the abstract: CEGIS, SyGuS, CDCL/CSP learning, and CEGAR already establish
"failed candidate -> verifier/counterexample -> refined search" as a core
formal-methods pattern. The paper's contribution is the narrower executable
artifact layer around that pattern: heterogeneous verifier traces, learned
residual rankers, and generated claim gates.

The accountability pass gives the venue-fit story: Model Cards, Datasheets, and
the NeurIPS reproducibility program ask authors to document intended use,
performance, provenance, limitations, code, and data. The claim compiler is a
partial executable answer to that agenda, not a replacement for human judgment.

## High-Value Insights

1. Exact verification is not just a better reward; it changes the failure mode.

Learned verifiers and reward models improve sampling systems, but the papers
repeatedly show calibration, over-optimization, and "right answer, wrong
reason" failure modes. Cobbe et al. show verifier selection improves GSM8K but
can degrade with too many completions because adversarial solutions start
fooling the verifier (`paper/sources/text/cobbe2021training.txt:502`,
`paper/sources/text/cobbe2021training.txt:512`). Stiennon et al. explicitly
warn that reward-model optimization has a limit beyond which evaluations become
less useful (`paper/sources/text/stiennon2020learning.txt:434`), and their
overoptimized samples become longer and lower quality
(`paper/sources/text/stiennon2020learning.txt:1996`).

CRLPLRIMES should say: exact verifier rewards do not remove all problems; they
move the main risks to coverage, search budget, grammar completeness, and
feature leakage. That is a more precise claim than "verifiers are better."

2. Process evidence matters even when final-answer accuracy looks similar.

Uesato et al. find outcome supervision can be close on final-answer accuracy,
while process feedback or a reward model emulating it is needed for low trace
error (`paper/sources/text/uesato2022solving.txt:83`,
`paper/sources/text/uesato2022solving.txt:950`). Lightman et al. make the same
case at step level: process supervision outperforms outcome supervision on MATH
and PRM800K supplies step labels (`paper/sources/text/lightman2023verify.txt:20`).

This supports the repo's certificate trace design. The paper should report not
only top-1 or verifier-call reductions, but also trace-level artifacts:
accepted/rejected rules, false-rejection counts, certificate reason codes, and
feature provenance.

3. Incorrect candidates are useful data, not noise.

V-STaR's key move is using both correct and incorrect generated solutions to
train a verifier with preference learning (`paper/sources/text/hosseini2024vstar.txt:20`,
`paper/sources/text/hosseini2024vstar.txt:522`). STaR improves by iterating on
self-generated rationales that lead to correct answers
(`paper/sources/text/zelikman2022star.txt:20`,
`paper/sources/text/zelikman2022star.txt:27`), but V-STaR shows the rejected side
of the sample distribution is valuable too.

CRLPLRIMES already records rejections. Treat those as a first-class dataset:
train or audit the proposer on "why rejected" rows, not only on accepted
certificates.

4. The closest structural relatives are formal-search systems, not RLHF.

AlphaGeometry's loop is neural auxiliary-construction proposal plus symbolic
deduction (`paper/sources/text/trinh2024alphageometry.txt:103`,
`paper/sources/text/trinh2024alphageometry.txt:218`). GPT-f uses generated proof
steps with a formal verifier, and emphasizes that proofs are quickly checked and
synthetic proof data can be generated (`paper/sources/text/polu2020generative.txt:33`,
`paper/sources/text/polu2020generative.txt:542`). AlphaGeometry2 strengthens the
same line with broader geometry language coverage and knowledge sharing between
search trees (`paper/sources/text/chervonyi2025alphageometry2.txt:23`,
`paper/sources/text/chervonyi2025alphageometry2.txt:878`).

The paper should foreground AlphaGeometry / GPT-f / AlphaGeometry2 before RLHF.
RLHF is a regularization and contrast class; formal verifier search is the
architectural family.

5. DeepSeek-R1 validates verifiable rewards at scale, but not our audit story.

DeepSeek-R1 avoids human reasoning traces, trains from final correctness, and
uses verifiable rewards where possible (`paper/sources/text/deepseek2025r1.txt:40`).
It also uses KL/reference controls inside GRPO
(`paper/sources/text/deepseek2025r1.txt:105`) and automatic evaluation for logic
datasets (`paper/sources/text/deepseek2025r1.txt:1019`).

This is strong contemporary support for "exact rewards when available." The
contrast is that DeepSeek couples verifiable rewards to massive sampling and
opaque infrastructure, while CRLPLRIMES makes every proposal, certificate, cost,
and provenance row auditable.

6. ARC is useful here as one diagnostic in the evidence portfolio.

The ARC Prize report says top private-set score rose from 33% to 55.5% using
deep-learning-guided program synthesis and test-time training
(`paper/sources/text/chollet2024arcprize.txt:17`). It warns that search-based
scores must be reported with compute budgets because more generated programs can
always raise the score (`paper/sources/text/chollet2024arcprize.txt:343`), and
estimates that 85% with a Greenblatt-style approach could require about 100M
programs per task (`paper/sources/text/chollet2024arcprize.txt:347`).

For CRLPLRIMES, matched-budget reporting is not optional. It is part of the
claim. ARC results should be written as one verifier-gate diagnostic inside the
multi-domain evidence portfolio.

7. Test-time adaptation is the most relevant near-term experimental extension.

Akyurek et al. show TTT can give large ARC gains, reaching 53.0% with an 8B model
and 61.9% when ensembled with program synthesis
(`paper/sources/text/akyurek2024surprising.txt:27`,
`paper/sources/text/akyurek2024surprising.txt:37`). The implementation uses
leave-one-out task construction, invertible transformations, per-task adapters,
and hierarchical voting (`paper/sources/text/akyurek2024surprising.txt:471`,
`paper/sources/text/akyurek2024surprising.txt:623`). The ARC Prize report also
says static inference-style transduction above 11% did not exist in that setting
(`paper/sources/text/chollet2024arcprize.txt:360`).

The realistic CRLPLRIMES variant is not an 8B LoRA pipeline, and the reason to
try it would not be leaderboard chasing. It would be a small per-instance update
to hypothesis-prior tables or rule-proposer logits using the current task's
train pairs, with the certificate gate unchanged, to test whether the substrate
supports task-local adaptation while preserving auditability.

8. Hypothesis objects should become library objects.

DreamCoder shows that learning reusable abstractions reduces search depth while
a recognition model reduces search breadth (`paper/sources/text/ellis2021dreamcoder.txt:205`).
Hypothesis Search makes the same cognitive move in a modern LLM form: generate
explicit hypotheses, compile to programs, verify on examples, then generalize
(`paper/sources/text/wang2024hypothesis.txt:20`,
`paper/sources/text/wang2024hypothesis.txt:29`). It also reports strong gains
over direct prompting on ARC-like and program-synthesis tasks
(`paper/sources/text/wang2024hypothesis.txt:35`,
`paper/sources/text/wang2024hypothesis.txt:441`).

This points to the cleanest Phase-5 research move: promote repeated successful
slot bindings or certified hypotheses into new templates. That is more aligned
with the substrate than adding a larger scorer.

9. Calibration is a central result, not a side metric.

Guo et al. show modern neural nets can become more accurate but less calibrated;
confidence stops tracking correctness (`paper/sources/text/guo2017calibration.txt:10`,
`paper/sources/text/guo2017calibration.txt:220`). Temperature scaling is a
surprisingly strong one-parameter fix (`paper/sources/text/guo2017calibration.txt:438`).
Cobbe et al. also show verifier overfitting manifests as overconfidence
(`paper/sources/text/cobbe2021training.txt:303`).

The paper's verifier-RL calibration section should stay prominent. It is one of
the clearest ways to show that exact rewards still need policy regularization
and calibration auditing.

10. KL/trust-region regularization is borrowed machinery; the novelty is where
it is applied.

TRPO derives robust policy improvement from KL trust regions
(`paper/sources/text/schulman2015trust.txt:203`,
`paper/sources/text/schulman2015trust.txt:237`). PPO operationalizes this with a
clipped surrogate objective that removes incentive for too-large policy-ratio
changes (`paper/sources/text/schulman2017proximal.txt:114`,
`paper/sources/text/schulman2017proximal.txt:122`). Ouyang et al. and Stiennon
et al. put PPO/KL-to-prior in the canonical RLHF pipeline
(`paper/sources/text/ouyang2022training.txt:112`,
`paper/sources/text/ouyang2022training.txt:481`,
`paper/sources/text/stiennon2020learning.txt:20`).

Do not present KL/entropy regularization as novel. Present "KL/entropy under an
exact verifier reward with schema-attested traces" as the contribution.

11. CEGIS is the direct ancestor of the certificate-gating loop.

Solar-Lezama describes Sketching as partial programs with holes, solved by a
SAT-based inductive synthesizer plus validation; the abstract explicitly says
CEGIS combines inductive synthesis with validation to automatically generate
test inputs and check the generated program against the specification
(`paper/sources/text/solarlezama2008sketching.txt:121`,
`paper/sources/text/solarlezama2008sketching.txt:132`). The body makes the
feedback loop even more explicit: validation checks a candidate, returns a
counterexample if it is wrong, and feeds that counterexample back so the next
candidate covers the corner case
(`paper/sources/text/solarlezama2008sketching.txt:893`,
`paper/sources/text/solarlezama2008sketching.txt:904`).

CRLPLRIMES should not imply it invented counterexample feedback. The right
claim is: it takes that loop shape and makes the logged verifier outcomes
usable by small residual rankers and by a paper claim compiler.

12. SyGuS names the grammar-constrained search setting.

Alur et al. frame synthesis as search over a permissible expression class, and
their CEGIS explanation treats the learner as searching candidate concepts
consistent with examples before a verification oracle checks correctness and
returns counterexamples (`paper/sources/text/alur2013sygus.txt:191`,
`paper/sources/text/alur2013sygus.txt:195`,
`paper/sources/text/alur2013sygus.txt:210`). This is the closest label for the
repo's grammar-plus-verifier setup.

The paper should use SyGuS to lower novelty pressure: the candidate language is
not an ad hoc trick. The new contribution is the cross-domain schema and
learned residual scoring around grammar-bounded candidates.

13. Conflict-driven learning is a stronger ancestor than our reason-coded rows.

GRASP treats conflicts as inevitable, analyzes their causes, backtracks
nonchronologically, records conflict causes, and preempts similar conflicts
later (`paper/sources/text/marquessilva1999grasp.txt:13`,
`paper/sources/text/marquessilva1999grasp.txt:16`). It also describes conflicts
as missing clauses in the problem specification and augments the problem with
conflict-induced clauses (`paper/sources/text/marquessilva1999grasp.txt:73`,
`paper/sources/text/marquessilva1999grasp.txt:83`). Bayardo and Schrag's
look-back work calls nogood-encoding clauses "reasons" and describes learning
schemes that record derived reasons for later search
(`paper/sources/text/bayardo1997csp.txt:99`,
`paper/sources/text/bayardo1997csp.txt:158`).

This is the clearest place to be conservative. CRLPLRIMES reason codes are not
learned clauses and do not have CDCL's sound pruning semantics. They are
training and audit features for budgeted residual ranking; the deterministic
verifier still owns admissibility.

14. CEGAR supports the "failed trace becomes refinement" framing.

Clarke et al. present an iterative abstraction-refinement methodology where
spurious counterexamples are analyzed and used to refine the abstract model
(`paper/sources/text/clarke2000cegar.txt:11`). They emphasize that when a
counterexample is an artifact of approximation, the abstraction must be refined
to eliminate that erroneous behavior
(`paper/sources/text/clarke2000cegar.txt:38`). Their algorithm identifies the
shortest invalid prefix, splits an abstract state, and notes that the coarsest
refinement problem is NP-hard, so a polynomial heuristic is used
(`paper/sources/text/clarke2000cegar.txt:76`,
`paper/sources/text/clarke2000cegar.txt:85`).

Use this as background, not as the main ancestor. CEGIS is closer because it is
candidate synthesis; CEGAR is useful for the broader pattern of using
counterexamples to refine the representation.

15. The claim compiler is an executable accountability artifact, but not a full
Model Card or Datasheet replacement.

Model Cards motivate documentation for intended use, performance
characteristics, evaluation context, and out-of-scope uses
(`paper/sources/text/mitchell2019modelcards.txt:7`,
`paper/sources/text/mitchell2019modelcards.txt:259`). Datasheets similarly
require motivation, composition, collection process, recommended uses, and
other provenance details, with an explicit transparency/accountability goal
(`paper/sources/text/gebru2021datasheets.txt:37`,
`paper/sources/text/gebru2021datasheets.txt:63`). Pineau et al. frame
reproducibility around code, data, a reproducibility challenge, and a checklist
(`paper/sources/text/pineau2021reproducibility.txt:41`,
`paper/sources/text/pineau2021reproducibility.txt:48`).

The executable claim compiler belongs in this lineage because it mechanically
checks evidence rows before prose can claim them. But Datasheets explicitly
warns that automated documentation can work against reflective documentation
goals (`paper/sources/text/gebru2021datasheets.txt:97`). The paper should say
the compiler enforces artifact support; it does not replace human scope,
ethics, intended-use, or limitation review.

16. Henderson et al. turns the Signal baseline gate into a literature-backed
requirement.

Henderson et al. show that RL results can differ drastically across seeds, and
that even averaging small numbers of trials can be misleading
(`paper/sources/text/henderson2018deep.txt:327`,
`paper/sources/text/henderson2018deep.txt:342`). Their recommendations stress
hyperparameter/baseline matching, codebase matching, fair comparisons, and many
trials with different random seeds
(`paper/sources/text/henderson2018deep.txt:482`,
`paper/sources/text/henderson2018deep.txt:499`).

This directly supports the current Signal boundary. Live neural-active
execution is claimable from telemetry, but comparative active-control
improvement needs baseline-matched episode summaries, variance accounting, and
clear environment/application scope.

## Per-Paper Use

| Paper | What It Gives Us | How To Use It |
| --- | --- | --- |
| Cobbe 2021 | Learned verifier ranks many sampled math solutions; selection helps but overfitting/adversarial samples appear. | Contrast learned verifier ranking with exact certificate gating; motivate calibration and sample-budget reporting. |
| Uesato 2022 | Outcome and process feedback can look similar on final answer but differ on reasoning trace quality. | Justify reporting traces and certificates, not only top-1. |
| Lightman 2023 | Step-level process supervision beats outcome supervision on MATH; PRM labels are costly but powerful. | Position certificates as machine-checkable process evidence. |
| Hosseini 2024 V-STaR | Incorrect generated solutions improve verifier training. | Preserve rejected-rule rows and use reason codes as training/audit data. |
| DeepSeek-R1 2025 | Verifiable rewards can drive large-scale reasoning RL without human traces. | Contemporary support for exact reward preference; contrast scale and auditability. |
| Christiano 2017 | Human preference reward models are the RLHF root. | Cite as contrast when no exact verifier exists. |
| Stiennon 2020 | Reward-model RL needs KL/entropy controls and can over-optimize. | Borrow regularization pattern; cite over-optimization risk. |
| Ouyang 2022 | SFT -> RM -> PPO became the canonical instruction-following pipeline. | Cite as background for PPO/KL machinery, not as a close structural relative. |
| AlphaGeometry 2024 | Neural proposer plus symbolic deduction engine. | Foreground as closest structural ancestor. |
| AlphaGeometry2 2025 | Search-tree knowledge sharing and broader formalization pipeline. | Use for "formal-search systems are scaling" context and limitations. |
| GPT-f 2020 | LM-generated proof steps checked by a formal verifier. | Historical ancestor for "model proposes, verifier checks." |
| DreamCoder 2021 | Learns reusable DSL abstractions and neural search priors. | Motivate library learning over certified hypothesis templates. |
| Hypothesis Search 2024 | Explicit hypotheses compiled to executable code and verified on examples. | Use as closest cognitive/hypothesis-layer relative. |
| Chollet 2019 | Intelligence as skill-acquisition efficiency; ARC as few-shot broad generalization. | Keep priors and experience explicit; avoid score-only framing. |
| ARC Prize 2024 | ARC SOTA is TTT/program-synthesis/search with compute budget sensitivity. | Honest context for the ARC diagnostic and evidence that compute-budget reporting is mandatory. |
| Akyurek 2024/2025 | Test-time training plus transformations/voting drives strong ARC gains. | Optional substrate test: small per-instance prior/proposer adaptation under the same audit rules. |
| Guo 2017 | Calibration can worsen even when accuracy improves; temperature scaling helps. | Anchor ECE/reliability-bin discussion. |
| Schulman 2017 PPO | Clipped policy updates stabilize policy gradient. | Cite as borrowed optimization method. |
| Schulman 2015 TRPO | KL trust regions ground conservative policy improvement. | Cite for trust-region intuition. |
| Zelikman 2022 STaR | Self-training loop over generated rationales that lead to correct answers. | Use as self-improvement lineage; V-STaR is the verifier-augmented successor. |
| Solar-Lezama 2008 Sketching | CEGIS couples inductive synthesis with validation-generated counterexamples. | Cite as the direct formal ancestor of the certificate-gating/rejection-feedback loop. |
| Alur 2013 SyGuS | Defines synthesis under a syntactic candidate grammar and semantic specification. | Use to name the grammar-plus-verifier setting; do not imply this part is novel. |
| Marques-Silva and Sakallah 1999 GRASP | Conflict analysis records conflict causes as learned clauses and enables nonchronological backtracking. | Use as a strong ancestor for "rejected candidates become reusable search information"; distinguish from our weaker reason-code learning. |
| Bayardo and Schrag 1997 CSP look-back | Nogood/reason learning and relevance-bounded learning improve SAT/CSP search. | Cite alongside GRASP for independent conflict-learning lineage. |
| Clarke 2000 CEGAR | Spurious counterexamples refine abstractions while keeping abstract state space small. | Use as broader counterexample-feedback background, secondary to CEGIS. |
| Pineau 2021 reproducibility | NeurIPS reproducibility program combines code policy, reproducibility challenge, and checklist. | Frame `make paper-ci` and the claim compiler as executable support for reproducibility disclosure. |
| Mitchell 2019 Model Cards | Structured model documentation covers intended use, evaluation, performance, and caveats. | Position trace schemas and generated claim blocks as executable model/reporting cards for this artifact. |
| Gebru 2021 Datasheets | Dataset documentation records motivation, composition, collection, use, maintenance, and limits. | Use as provenance lineage; keep the caveat that automation does not replace reflective documentation. |
| Henderson 2018 Deep RL That Matters | RL claims are sensitive to seeds, codebases, hyperparameters, environment details, and baselines. | Justify the Signal limitation and the required active neural vs heuristic baseline gate. |

## Concrete Paper Edits Suggested

1. Rename the main related-work contrast from "learned verifiers" to
   "verifier-bounded search and reward." That better covers AlphaGeometry,
   GPT-f, DeepSeek-R1, PRMs, V-STaR, and ARC program synthesis.

2. Add one sentence in the abstract or introduction:

   "The contribution is an auditable verifier-bounded learning protocol for
   domains where exact checks are cheap
   enough to shape search."

3. In the limitations section, add a sharper exact-verifier limitation:

   "Exact rewards remove reward-model misgeneralization, but they do not address
   search coverage, grammar expressiveness, verifier-cost scaling, or confidence
   calibration."

4. In the experiment section, keep matched-budget and latency fields near every
   result table. The ARC Prize report makes this a benchmark-level requirement,
   not a nice-to-have.

5. State explicitly that the experiment portfolio is the claim: prime-gap,
   grid-path, delayed-claim, ARC, and program synthesis are different
   instantiations of one verifier-bounded protocol.

6. In the roadmap, prioritize these over broader web/product work:

   - Rejected-candidate learning from certificate reason codes.
   - Per-instance adaptation of small hypothesis priors.
   - Library learning that promotes repeated certified slot bindings.
   - More calibration audits across non-prime domains.

7. Add one paper-construction claim that is explicitly scoped:

   "The claim compiler converts parts of the Model Card / Datasheet /
   reproducibility-checklist agenda into executable artifact gates: it checks
   that numeric claims, trace provenance, environment identity, and limitations
   are generated from repository artifacts before paper compilation. It does
   not replace human review of intended use, interpretation, or ethical scope."

8. Keep the Signal next gate literature-backed:

   "Live neural-active execution is a current artifact claim. A comparative
   active-control claim requires baseline-matched episodes, multiple seeds or
   windows where applicable, utility/regret/damage/death summaries, and replay
   traces sufficient to diagnose environment and implementation variance."

## Strongest NeurIPS/ArXiv Framing

The paper should be assessed against this thesis:

> In verifier-equipped domains, learning should be evaluated as a controlled
> interaction between proposal, exact checking, residual ranking, and traceable
> provenance. CRLPLRIMES demonstrates that this interaction can be made
> schema-attested and reproducible across heterogeneous toy-to-small domains,
> while preserving honest budget and calibration measurements.

That is arXiv-quality if the claims are kept narrow and every reported number is
trace-backed. For NeurIPS, the likely weak points are scale, novelty against
AlphaGeometry-style systems, and breadth of empirical comparison. The best path
to strengthen it is not a bigger ARC score. It is one clean cross-domain result
that only this substrate makes easy, such as rejected-candidate learning or
library promotion from certified hypotheses.
