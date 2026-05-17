# Reference Synthesis: How This Paper Sits in the Current Literature

**Purpose.** This is the working bibliography + analysis the paper draws from. Each entry has: full citation, verbatim abstract or near-verbatim summary, one-paragraph positioning against this project's claims, and (where applicable) ideas worth borrowing into the substrate. The Related Work section in `paper.typ` should be regenerated from this document by the `cenetex/agent` flow.

The headline analytical claim: **our paper's closest active relative is AlphaGeometry; our closest *contemporary methodological* relative is DeepSeek-R1's RLVR; our closest *cognitive* relative is Hypothesis Search.** All three confirm the architectural premise (exact verifiers + neural proposers are a real design point). What we add over them is engineering discipline — schema-attested provenance, matched-budget oracle baselines, tiny C11 substrate, paper-as-CI-artifact — applied to *heterogeneous* tasks.

Last updated: 2026-05-17.

---

## Group A — Verifier-gated reasoning (the line we are explicitly arguing against the centralized form of)

### A1. Cobbe et al. 2021 — *Training Verifiers to Solve Math Word Problems* (GSM8K)
- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman. arXiv:2110.14168.
- **What they do.** Introduce GSM8K (8.5K grade-school math word problems). Train a *learned verifier* to rank multiple LLM-sampled solutions; pick the top-ranked one.
- **Relation to us.** Founding paper of the "verifier-guided decoding" line. Their verifier is learned (a fine-tuned LM head). Ours is exact (algorithmic, certificate-backed). Their architecture: sample → verify. Ours: verify → score the survivors.
- **Idea worth borrowing.** Their per-sample verifier score is a `crlp.compare.v1`-shaped row before our schema existed; the format invariant of "candidate row + verifier outcome + cost" is reusable.

### A2. Lightman et al. 2023 — *Let's Verify Step by Step* (PRM800K)
- Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe. arXiv:2305.20050.
- **What they do.** Compare *process supervision* (label every reasoning step) to *outcome supervision* (label only the final answer) for training reward models. PRM-trained models reach 78% on a MATH subset. Release PRM800K, 800K step-level human labels.
- **Relation to us.** Their PRM is the canonical learned-step-verifier. Our `NNFeatureProvenance` taxonomy distinguishes structural priors, state observations, verifier-derived features, and gate-derived features — the same distinction PRM literature gestures at informally, formalized as a type-checkable invariant.
- **Idea worth borrowing.** Active learning to choose which steps get human supervision. Our analog would be active learning to choose which *hypotheses* get certificate checks under tight budgets.

### A3. Uesato et al. 2022 — *Solving math word problems with process- and outcome-based feedback*
- Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, Irina Higgins. arXiv:2211.14275 (DeepMind).
- **What they do.** Comprehensive comparison of process vs outcome supervision on GSM8K. Reasoning error among correct solutions: 14.0% → 3.4% with process supervision; final-answer error 16.8% → 12.7%.
- **Relation to us.** Same conceptual axis as Lightman; their finding that *outcome supervision is competitive on final-answer accuracy but worse on reasoning quality* is structurally similar to our finding that *gated top-1 saturates while raw NLL keeps drifting* (Axis 3's ECE result; ARC's calibration outlier). The mechanism is different (we're not training a step verifier) but the failure mode (right answer for the wrong reason) is the same warning.

### A4. Hosseini et al. 2024 — *V-STaR: Training Verifiers for Self-Taught Reasoners*
- Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, Rishabh Agarwal. arXiv:2402.06457.
- **What they do.** STaR iterates self-improvement on correct solutions; V-STaR also uses *incorrect* solutions to train a DPO verifier. Iterating both yields better reasoners + verifiers. 4–17% accuracy gains on code and math benchmarks vs prior STaR.
- **Relation to us.** The "both correct and incorrect solutions train the verifier" idea is a learned-reward variant of what our `crlp.program_proposer.summary.v1` schema does naturally: every accepted rule has a certificate, every rejected rule has a `proof_kind` and a `reason` recorded. The closed-loop residual rule policy is the structural analog of V-STaR's iterative loop; ours has the exact-verifier property they don't.

### A5. DeepSeek-AI et al. 2025 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- DeepSeek-AI (team). arXiv:2501.12948 / Nature 2025.
- **What they do.** Train reasoning models with pure RL on **verifiable correctness checks** ("does this compile? does this math evaluate? does this output match the required format?") — not on learned reward models. Explicitly motivated by "neural reward models are susceptible to reward hacking during large-scale RL." Names this *RLVR* (Reinforcement Learning from Verifiable Rewards).
- **Relation to us.** *The most important contemporary paper to cite.* DeepSeek-R1 makes the same architectural argument we make — exact verifiers should be preferred to learned reward models when available — at hundred-billion-parameter scale. Our paper is the small-scale, audit-clean version of the same thesis. The contrast point: they couple it to massive CoT-style sampling; we couple it to deterministic rule synthesis + tiny scorer.
- **Idea worth borrowing.** Reward hacking under learned reward models is an *empirical* finding at their scale that we should cite as motivating evidence for the verifier-first ordering, not just architectural intuition.

### A6. Christiano et al. 2017 — *Deep RL from Human Preferences*
- Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei. arXiv:1706.03741.
- **What they do.** Pair RL with learned reward from human pairwise preferences. Atari games and simulated robot locomotion solved with <1% of interactions labeled.
- **Relation to us.** Founding paper of RLHF. The "learned reward + RL" stack we're arguing the substrate doesn't need when an exact verifier exists. Cite as the canonical alternative; our Axis 3 (KL-to-prior + entropy regularization on the exact-reward gradient) is a direct application of the regularization pattern this paper made standard.

### A7. Stiennon et al. 2020 — *Learning to summarize from human feedback*
- Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano. arXiv:2009.01325.
- **What they do.** RLHF on summarization. Establishes the now-canonical "KL-to-prior + entropy bonus" regularization on the reward-model gradient. 1.3B InstructGPT-precursor model outperforms 175B GPT-3 on summary preference.
- **Relation to us.** Axis 3 of our paper (the calibration result) is "the Stiennon objective applied to *exact* rewards instead of preference rewards." Cite as the methodological precedent. We are not claiming the regularization mechanism is novel; we are claiming its applicability when the reward is exact-by-construction.

### A8. Ouyang et al. 2022 — *Training language models to follow instructions with human feedback (InstructGPT)*
- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe. arXiv:2203.02155.
- **What they do.** RLHF pipeline made canonical: SFT → reward model on preferences → PPO with KL-to-SFT. 1.3B model > 175B GPT-3 on instruction-following human preference.
- **Relation to us.** Cite alongside Stiennon as the canonical RLHF reference. Our Axis 3 protocol is the same shape applied to verifier rewards.

---

## Group B — Neuro-symbolic search with formal verifiers (the closest structural ancestors)

### B1. Trinh, Wu, Le, He, Luong 2024 — *AlphaGeometry: Solving olympiad geometry without human demonstrations*
- Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, Thang Luong. *Nature*, 2024.
- **What they do.** Neural language model proposes geometric auxiliary constructions; a deterministic symbolic deduction engine verifies whether the resulting figure proves the goal. Bootstrapped on 100M synthetic theorems. 25/30 olympiad geometry problems, matching average IMO gold medalist (25.9).
- **Relation to us.** *Closest active structural relative.* Same architectural shape — neural proposes, formal engine decides, search interleaves both. Differences: their proposer is a transformer trained on 100M synthetic proofs; ours is a hundreds-of-parameter MLP. Their domain is Lean-checkable geometry; ours is a portfolio of verifier-equipped task domains. *The protocol is the same; the scale is fundamentally different.* This is the paper to position against.
- **Idea worth borrowing.** Synthetic-data bootstrapping. Their 100M synthetic theorems are how they avoid human-curated proofs; we could synthesize ARC-like tasks to bootstrap the hypothesis-induction learned prior the way they bootstrap geometric tactics.
- **Followup.** AlphaGeometry2 (arXiv:2502.03544) — gold-medalist-level performance. Cite for completeness.

### B2. Polu & Sutskever 2020 — *Generative Language Modeling for Automated Theorem Proving (GPT-f)*
- Stanislas Polu, Ilya Sutskever. arXiv:2009.03393.
- **What they do.** GPT-style language model proposes Metamath tactics; a symbolic checker verifies each proof step. First deep-learning system to contribute novel proofs accepted into the main Metamath library.
- **Relation to us.** Earliest "LM proposes, formal verifier decides" paper in this lineage. Should be cited as the historical antecedent of AlphaGeometry / our work. The point we make against it: their LM has to learn the *entire* deductive substrate from data; our substrate carries the grammar explicitly and the scorer only learns to *rank* within it.

### B3. Ellis et al. 2021 — *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning*
- Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sable-Meyer, Luc Cary, Lucas Morales, Luke Hewitt, Armando Solar-Lezama, Joshua B. Tenenbaum. arXiv:2006.08381.
- **What they do.** Wake-sleep neural-guided enumeration over a domain-specific language. Learns reusable abstractions ("rediscovers" functional programming primitives, vector algebra, classical mechanics laws). Library-learning across creative tasks (drawing, scenes) and inductive programming.
- **Relation to us.** *Closest cognitive relative of the `NNHypothesis` layer.* Their template = our NNHypothesisTemplate; their abstraction = our compiled NNProgram; their neural search guidance = our learned hypothesis slot prior. Differences: DreamCoder is GPU Python, learns abstractions over months from scratch; our substrate is C11, instantiates templates inline, uses certificates rather than empirical support.
- **Idea worth borrowing.** *Library learning is the next research move.* DreamCoder grows its DSL by compressing repeated patterns in solved tasks; our hypothesis registry is currently hand-authored. A wake-sleep loop that promotes successful slot bindings to *new templates* would be a natural Phase-5 direction.

### B4. Wang et al. 2024 — *Hypothesis Search: Inductive Reasoning with Language Models*
- Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, Noah D. Goodman. ICLR 2024 / arXiv:2309.05660.
- **What they do.** LLM proposes natural-language hypotheses about an inductive task; converts each hypothesis to a Python program; runs the program on training examples to verify; selects the consistent one. Tested on ARC, 1D-ARC, SyGuS, List Functions. 33% on a 100-problem ARC subset (with human in the loop), 30% automated, vs 17% direct prompting.
- **Relation to us.** *Naming convergence on "hypothesis."* Their hypothesis is a free-form NL description compiled to Python by an LLM; ours is a typed slot binding compiled to a canonical `NNProgram` by a deterministic compiler. Their search is over LLM samples; ours is over templates + evidence-derived bindings. The conceptual move is the same; our implementation is verifier-first and audit-clean, theirs is sample-and-test.
- **Idea worth borrowing.** Their "summarize hypotheses into a smaller set" filtering step is a real cost-reduction technique. Our analog: hypothesis canonical_hash deduplication is already implemented but we could add a *clustering* step that summarises multiple bound hypotheses into a fewer-distinct-class summary before compilation.

---

## Group C — ARC-specific (the benchmark we partially address)

### C1. Chollet 2019 — *On the Measure of Intelligence*
- François Chollet. arXiv:1911.01547.
- **What they do.** Introduces the Abstraction and Reasoning Corpus (ARC). Formal definition of intelligence as *skill-acquisition efficiency*. Argues benchmark scores alone don't measure intelligence because priors and training data can buy arbitrary skill.
- **Relation to us.** Defines the benchmark we partially use. Our position: *the audit machinery is itself a step toward Chollet's intelligence definition* — by tracking which features are state-observations vs structural-priors vs gate-derived, we make explicit what priors we're using. We are not claiming the benchmark is solved; we are claiming the framework makes prior-budgeting visible.

### C2. Chollet, Knoop, Kamradt, Landers 2024 — *ARC Prize 2024: Technical Report*
- François Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers. arXiv:2412.04604.
- **What they do.** SOTA on the ARC private evaluation set rose from 33% to 55.5% in 2024. Top techniques: test-time training (TTT) + deep-learning-guided program synthesis. ARChitects won the open-source prize at 53.5%; MindsAI scored 55.5% but did not open-source.
- **Relation to us.** *The competitive baseline we should explicitly position against.* Our public-corpus floor is 24/400 ≈ 6% on the training split; the SOTA is 55.5% on the (different but related) private evaluation. We are not in this race — our contribution is the substrate, not the score. Cite this paper to make the gap honest.
- **Idea worth borrowing.** Test-time training as a mechanism — *update model weights at inference using the train pairs of the current task.* This is structurally compatible with our hypothesis induction layer (induce hypothesis on test task's train pairs, certify against those pairs, apply to test) but with neural-weight adaptation. Worth exploring whether a small TTT step on the hypothesis prior at inference helps.

### C3. Akyürek et al. 2024 — *The Surprising Effectiveness of Test-Time Training for Abstract Reasoning*
- Ekin Akyürek, Mehul Damani, et al. arXiv:2411.07279 (MIT).
- **What they do.** TTT with in-context ARC examples — up to 6× higher accuracy than fine-tuned baselines. 53% public-validation with an 8B-parameter LM, 61.9% ensembled with program synthesis (matches average human). Three crucial components: (1) initial fine-tuning on similar tasks, (2) auxiliary task format + augmentations, (3) per-instance training.
- **Relation to us.** Direct competitor on ARC scores; far above our 24/400. *Our argument is that their result depends on a 8B LM + test-time gradient updates, which presumes per-task compute they don't audit.* The substrate contribution we make — schema-enforced feature provenance, certificate-gated rules — is orthogonal to their approach; either could in principle be combined.
- **Idea worth borrowing.** "Per-instance training" — they train on augmentations of the *current task's* train pairs. Our `move_to_anchor` apply-time scene re-derivation is a no-gradient form of per-instance specialization; a learned form would be the natural next step.

---

## Group D — Calibration (Axis 3 grounding)

### D1. Guo, Pleiss, Sun, Weinberger 2017 — *On Calibration of Modern Neural Networks*
- Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. ICML 2017 / arXiv:1706.04599.
- **What they do.** Modern neural networks are systematically miscalibrated (confidence > accuracy). Temperature scaling — a single-parameter Platt-scaling variant — fixes it cheaply. Identifies depth/width/weight-decay/BatchNorm as calibration-degrading factors.
- **Relation to us.** Our Axis 3 reports `T=1.5` recovers strict-RL calibration; `T=1.0` suffices with KL/entropy regularization. The "single scalar to recover calibration after a training transformation degrades it" intuition is Guo's. Our ECE and reliability_bins schemas use the bin definitions this paper popularized.

### D2. Naeini, Cooper, Hauskrecht 2015 — *Obtaining Well Calibrated Probabilities Using Bayesian Binning (ECE)*
- AAAI 2015. ECE definition.
- **Relation to us.** Definitional reference for the ECE metric we report in every `crlp.compare.v1` row. Cite once in the methodology.

---

## Group E — RL with regularization (Axis 3 protocol family)

### E1. Schulman et al. 2017 — *Proximal Policy Optimization Algorithms (PPO)*
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. arXiv:1707.06347.
- **Relation to us.** Our strict and KL-regularized verifier RL branches use the same gradient style PPO uses. Cite in §3 method as the canonical reference for KL-bounded policy gradient.

### E2. Schulman et al. 2015 — *Trust Region Policy Optimization (TRPO)*
- John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, Philipp Moritz. ICML 2015 / arXiv:1502.05477.
- **Relation to us.** Predecessor of PPO. Cite for completeness on the KL-bound-as-trust-region intuition.

---

## Group F — Self-training / proposer-improvement loops (related search machinery)

### F1. Zelikman et al. 2022 — *STaR: Self-Taught Reasoner*
- Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman. arXiv:2203.14465 / NeurIPS 2022.
- **What they do.** Bootstrap reasoning by filtering self-generated solutions on outcome correctness, fine-tuning on the survivors. Iterate.
- **Relation to us.** Founding paper of "iterate self-improvement on verifier-correct samples." V-STaR (A4) is the verifier-augmented successor. Our closed-loop residual rule policy plays an analogous role: certify a rule, accept it, retrain the proposer on the accepted set, propose more.

---

## What's missing from the current draft's Related Work

The placeholder in `paper.typ` §Related Work lists Lightman, Cobbe, Hosseini, Trinh, Stiennon, Ouyang. **Six gaps to fill:**

1. **DeepSeek-R1 (A5).** The most contemporaneous and structurally aligned paper. Cite as confirmation of the verifier-over-learned-reward thesis at scale.
2. **AlphaGeometry (B1).** Closest structural ancestor. Currently mentioned only in commentary, not formally cited.
3. **GPT-f / Polu & Sutskever (B2).** Historical antecedent of the LM-proposes-formal-verifier-decides pattern.
4. **DreamCoder (B3).** Closest cognitive relative of the hypothesis layer.
5. **Hypothesis Search (B4).** Naming overlap; structurally we are doing the verifier-first version of their sample-and-test loop.
6. **ARC Prize 2024 Technical Report (C2) + Akyürek TTT (C3).** Required to honestly position our 24/400 floor against the 55.5% SOTA. The honest framing is that our ARC contribution is substrate-level, not score-level.

## Citation budget for the paper

A reasonable Related Work section runs 1.5–2 pages, citing 15–20 papers. The hierarchy:

- **Foreground (3 papers, one paragraph each):** AlphaGeometry, DeepSeek-R1, DreamCoder. The structural ancestors against which we have a real architectural argument.
- **Mid-ground (5–6 papers, grouped paragraphs):** Lightman, Uesato, Cobbe, V-STaR, Hypothesis Search. The verifier-guided line in NLP; we are the audit-clean small-scale variant.
- **Background (4–5 papers, one paragraph):** Ouyang, Stiennon, Christiano, Schulman PPO. RLHF / regularized policy gradient. Axis 3 borrows the regularization mechanism.
- **Benchmark context (3 papers):** Chollet *Measure*, ARC Prize 2024, Akyürek TTT. We are not in this race; we say so.
- **Calibration (1 paper):** Guo et al. temperature scaling.

## BibTeX

Stored at `paper/references.bib` (sibling file). Use `#bibliography("references.bib")` in `paper.typ` once the Related Work section is rewritten by the agent.

## Ideas this synthesis surfaces for the substrate

In rough order of cost vs payoff:

1. **Library learning over the hypothesis registry** (DreamCoder pattern). Promote successful slot bindings to first-class templates. Cheap engineering; the registry already exists.
2. **Per-instance test-time adaptation** of the hypothesis prior (Akyürek pattern). Use the test task's train pairs to update slot-prior log-odds tables at inference. Medium cost; the prior storage exists.
3. **Synthetic-data bootstrapping** for hypothesis-induced transforms (AlphaGeometry pattern). Generate ARC-shape synthetic tasks; certify against grammar; use as training data for the slot prior. Medium cost; tooling does not exist.
4. **Active selection of which hypotheses get certificate-checked** under tight budgets (PRM800K pattern). Choose the next hypothesis to verify by maximum expected information. Substrate change; useful when verifier cost dominates.
5. **NL-hypothesis bridge** (Hypothesis Search pattern). Allow templates to be authored in natural language and compiled to slot specs. Bigger lift; requires an LM in the loop. Probably not worth it.
