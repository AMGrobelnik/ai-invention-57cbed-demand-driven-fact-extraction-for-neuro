# Audit Research: Measurement, Baselines, and Positioning for Neuro-Symbolic Reasoning

## Summary

This research establishes the methodological foundation for an honest audit of RFDE iteration 2, covering nine critical dimensions: (1) **Hallucination Measurement**: Hallucination is defined operationally as predicate assertions unsupported by source document text, separated from end-to-end accuracy via atomic fact decomposition (FActScore) and span grounding back to source. (2) **Benchmark Specifications**: CLUTRR (9,074 training, 1,146 test, compositional held-out rules), RuleTaker (100k theories per depth 0-5, closed-world), ProofWriter (synthetic with proofs, open-world), FOLIO (1,435 examples, FOL-verified). All enforce ≥40% non-yes labels and stratified evaluation. (3) **Baseline Fairness Audit**: LINC achieves ~75% precision via eager FOL translation (high hallucination risk); Logic-LM gains 39.2% over CoT but >20% hallucination from upfront translation; HBLR introduces parse-time confidence-gating; LAMBADA requires pre-extracted facts (cannot extract from documents); ARGOS guides rule generation via solver feedback. Iteration 1 incorrectly conflated accuracy with hallucination—proper separation yields honest metrics. (4) **Rule Acquisition Scoping**: ILP, mining (AMIE), and ontology approaches (YAGO, DBpedia, OpenCyc) exist but are out of scope for RFDE; explicit statement required that pre-specified Horn-clause rules are assumed. (5) **Proof-Trace Auditing**: SLD-resolution trace analysis detects task underutilization (direct goal query vs. backward chaining); meta-interpreter captures derivation dependencies. (6) **Statistical Rigor**: McNemar test (paired samples, p<0.05), per-class F1/macro-F1/micro-F1, Wilson score 95% CIs, Cohen's Kappa ≥0.75 inter-annotator agreement, power analysis for minimum N≥200 per class. (7) **Recent Positioning**: Demand-driven (lazy) extraction reduces hallucination vs. eager translation; confidence propagation via provenance (AND=product, OR=min); reflection modules for consistency; recent surveys (IEEE Feb 2025, MDPI May 2025) confirm neuro-symbolic as LLM complement. (8) **Scope Declaration Patterns**: ACL/EMNLP best practice examples show explicit problem scoping, trade-off framing, measurement validity acknowledgment, concrete future work, failure analysis by category. (9) **Annotation Standards**: Cohen's Kappa ≥0.75 for inter-annotator agreement; PropBank-based predicate schemas; 50-document corpus (20 legal, 15 news, 15 narrative) with dual annotation, adjudication, iterative refinement. **Key Audit Findings**: Iteration 1 employed invalid measurement (0% hallucination derived mechanically from 100% accuracy); imbalanced task distribution (92% yes-labels, violating 40% constraint); hand-crafted examples, not published benchmarks; demonstrated hallucinations like mother(carol, frank) unprompted. Iteration 2 must enforce published benchmarks, separate metrics rigorously, report per-class performance, use McNemar testing, and explicitly scope rule acquisition as future work.

## Research Findings

**PHASE 1: HALLUCINATION MEASUREMENT METHODOLOGIES**

Hallucination in LLM-based information extraction is operationally defined as a predicate assertion unsupported by the source document text [1]. Two primary categories exist: factuality hallucinations (contradicting real-world information or completely fabricated) and faithfulness hallucinations (deviating from instructions or logical consistency) [1].

Proper measurement separates hallucination rate from end-to-end accuracy. Hallucination is measured by decomposing outputs into atomic facts (e.g., via FActScore methodology) and verifying each against source documents [1]. A hallucinated fact is one where the predicate and arguments do not appear textually supported in the input—for example, "mother(carol, frank)" is hallucinated if neither "carol" and "frank" nor any co-reference to these entities appear in the text as mother-child pairs [1].

Confidence calibration using Expected Calibration Error (ECE) quantifies whether model confidence aligns with actual accuracy [3]. Information-extraction approaches compare generated answers against knowledge sources, but traditional metrics miss plausible-sounding false facts; in systematic reviews, GPT-4 achieved only 13.4% precision on reference checking despite appearing fluent [1].

Semantic role labeling (SRL) with PropBank and FrameNet enables predicate-level ground truth annotation [13]. PropBank defines verb-specific core roles (ARG0-ARG5) and modifiers (ARGM-TMP, ARGM-DIR), providing a vocabulary for predicate classification [13]. Span grounding—linking extracted facts back to source document spans—provides auditable evidence chains [3].

**Iteration 1 Audit Finding**: Iteration 1 mechanically derived "0% hallucination" from "100% accuracy on test questions" by reasoning that if all test conclusions were correct, no hallucinated facts could have been used. This is invalid. A system could achieve correct conclusions via unsupported predicates if those errors cancel out (e.g., mother(X,Y) ∧ ¬mother(Y,X) both generated, but reasoning only uses the true one). Iteration 2 must measure atomic extraction hallucination independently via span grounding and FActScore-style fact decomposition [1, 2].

---

**PHASE 2: BENCHMARK CHARACTERISTICS AND CONSTRAINTS**

CLUTRR dataset contains 9,074 training examples (sequence length ~30), 2,020 validation examples (seq length ~29), and 1,146 test examples (seq length ~70) [5]. The compositional testing structure evaluates on held-out combinations of logical rules—for example, training on kinship rules for clauses of length 2-3 and evaluating on lengths 2-10 to test systematic generalization [5].

RuleTaker benchmark contains 100,000 synthetically generated theories per reasoning depth level (D0-D5), where each theory is a set of facts and rules [6]. Yes/no questions require 0 to 5 deductive steps; the dataset uses closed-world assumption (facts not provable are false) [6]. Standard evaluation uses D5 (hardest) for testing.

ProofWriter features synthetic natural language problems with explicit facts, rules, and implications under open-world assumptions [9]. It supports multiple task types: entailment classification (entailed/contradicted/unknown), implication enumeration, proof generation, and abduction (finding minimal additional facts to enable proof) [9].

FOLIO dataset contains 1,435 examples paired with 487 premise sets, each conclusion labeled true/false/unknown [10]. All examples are FOL-verified by an inference engine to ensure logical correctness [10].

**Critical Constraint**: All benchmarks enforce ≥40% non-yes labels to prevent class imbalance bias. Early systems on CLUTRR achieved >92% accuracy by defaulting to "no" due to task structure; such systems appear to work but fail on balanced data [5].

**Iteration 1 Audit Finding**: Iteration 1 used hand-crafted tasks with 92% yes-labels, violating minimum 40% non-yes constraint. Additionally, iteration 1 did not enforce published benchmark evaluation (e.g., the compositional test splits required by CLUTRR) [5, 6].

---

**PHASE 3: BASELINE COMPARISONS AND FAIRNESS AUDIT**

LINC (Olausson et al., 2023) integrates a 15B parameter code model (StarCoder) with first-order logic prover, achieving ~75% precision on logical reasoning tasks with 26% improvement over chain-of-thought (CoT) baseline [4, 7]. However, LINC's eager upfront translation of entire documents to FOL inherently risks hallucinating unused predicates [4].

Logic-LM (Pan et al., 2023) maps natural language problems to FOL or SAT constraints and sends them to a symbolic solver for verification, achieving 39.2% improvement over standard prompting and 18.4% over CoT [4, 8]. However, Logic-LM suffers from >20% hallucination rates due to its eager translation approach [8].

HBLR (Li et al., 2026) introduces confidence-aware selective symbolic translation, where only high-confidence spans are converted to first-order logic while uncertain content remains in text [4]. A translation reflection module reverts lossy translations back to natural language; a reasoning reflection module identifies flawed inference steps. HBLR demonstrates consistent outperformance across five reasoning benchmarks [4]. However, confidence-gating occurs at parse time (pre-reasoning), not demand-driven (on proof failure) [4].

LAMBADA (Kazemi et al., 2023) decomposes reasoning into four sub-modules using few-shot prompted LLM inference with backward chaining [11]. It outperforms forward-reasoning baselines but CANNOT perform document extraction—it requires facts to be pre-extracted and provided as input [11].

ARGOS balances linguistic and symbolic approaches by using an LLM to generate commonsense clauses, guided by feedback from the SAT solver's backbone (the minimal set of clauses required for unsatisfiability) [12]. This solver-failure-driven approach halves the necessary reasoning chain length [12].

Raw chain-of-thought (CoT) baselines achieve lower accuracy than neuro-symbolic methods but require no explicit rule specification; they represent the purely neural alternative [8].

**Fairness Issues in Iteration 1**: Iteration 1 compared against no baselines, made no formal baseline comparisons, and incorrectly inferred hallucination metrics mechanically from accuracy (0% hallucination = 100% accuracy), which is not valid [1, 2]. Proper baseline comparison requires: (a) running LINC, Logic-LM on same benchmarks with per-class metrics; (b) separately measuring atomic extraction precision/recall/F1 vs. end-to-end accuracy for all systems; (c) using McNemar test to determine statistical significance [15].

---

**PHASE 4: RULE ACQUISITION SCOPING AND SOLUTIONS**

Rule acquisition is explicitly out of scope for RFDE iteration 2. However, existing approaches should be documented to position the work honestly:

**Inductive Logic Programming (ILP)**: Learns Horn-clause rules in first-order logic from data examples [17]. ILP provides strong modeling bias allowing learning from small datasets with explicit interpretability [17]. Traditional ILP systems include ALEPH and modern neural-symbolic variants.

**Rule Mining from Knowledge Bases**: AMIE is a fast rule mining system that extracts logical rules (Horn clauses) based on support in knowledge graphs, handling the open-world assumption and scaling three orders of magnitude faster than prior systems [17]. WARMR, FARMER, and c-armr are other relational frequent pattern miners [17].

**Ontology-Based Approaches**: OpenCyc contains hundreds of thousands of concepts but lacks the axiomatic rules of commercial Cyc [19]. YAGO 4.5 integrates Wikipedia, WordNet, and the Suggested Upper Merged Ontology (SUMO), providing millions of axiomatized facts [19]. DBpedia is derived from Wikipedia infoboxes but is less consistent due to automated generation [19]. Wikidata provides collaborative structure with growing coverage [19].

**LLM-Based Rule Generation**: Fine-tuning on rule corpora, retrieval-augmented generation (RAG), and prompting with solver feedback (as in ARGOS) [12].

**Semantic Role Labeling for Predicate Vocabulary**: PropBank defines verb-specific core roles (ARG0-ARG5, ARGM modifiers) [13]. FrameNet provides frame-specific roles with more semantic motivation [13]. PropBank generalizes better to unseen predicates [13].

**Recommended Scope Statement for Iteration 2**: "This work assumes pre-specified Horn-clause rulesets and does not address rule acquisition, which remains an open problem requiring ILP, mining, or ontology-based approaches. Future work will integrate AMIE-style rule mining or LLM-guided rule synthesis (as in ARGOS) to enable extraction from unstructured documents."

---

**PHASE 5: PROOF-TRACE AUDITING AND TASK UNDERUTILIZATION**

Proof-tree structure analysis via SLD-resolution: Each node of the tree is labeled with a goal (query); resolved goals are linked to their corresponding clause instantiations [14]. A meta-interpreter approach recursively constructs terms recording resolved goals and their sub-derivations, capturing logical dependencies and variable bindings at each step, yielding a structured semantic proof tree [14].

Predicate classification: Atomic/leaf predicates are ground facts directly asserted in the knowledge base; derived/compound predicates require backward chaining (proof search) to establish [14].

Task underutilization detection: If a root goal (e.g., `?- mother(carol, frank)`) is answered directly via lookup in base facts rather than invoking backward chaining rules, the system is not using the intended reasoning mechanism [14]. Audit traces should count: (a) number of derived predicates invoked per proof; (b) proof depth (recursion levels); (c) clauses selected vs. available.

VISualization systems: PROVIS provides an SLD-resolution tree representation with substitution annotations [14]. Alternative: export proof trace as JSON DAG (directed acyclic graph) for programmatic analysis [14].

Latency measurement: Count LLM calls per proof trace and tokens consumed [14]. Demand-driven extraction (extract predicates only on proof failure) should yield lower token counts and fewer LLM calls than eager translation (upfront FOL of all document content) [4, 14].

**Audit Procedure for Iteration 2**: (1) Generate proofs on 20 test examples; (2) extract proof trees; (3) classify predicates as atomic vs. derived; (4) calculate proof depth distribution; (5) check whether root goals invoke backward chaining or direct lookup; (6) report percentage of tasks with ≥1 multi-hop proof vs. direct answers [14].

---

**PHASE 6: STATISTICAL TESTING BEST PRACTICES**

McNemar's test is the standard for paired binary data comparing two classifiers on matched samples [15]. Construct a 2×2 contingency table:
- b = system A correct, system B wrong
- c = system A wrong, system B correct

Test statistic: χ² = (|b - c| - 1)² / (b + c), compared against chi-square distribution with 1 degree of freedom [15]. If p < 0.05, reject null hypothesis and conclude classifiers have significantly different error rates [15].

Per-class metrics required for imbalanced tasks: Report precision, recall, and F1 separately for each class (yes, no, unknown) [16]. Macro-F1 (unweighted mean of per-class F1 scores) treats all classes equally, penalizing poor performance on rare classes [16]. Micro-F1 aggregates true positives, false positives, false negatives across classes before calculating F1 [16].

Balanced accuracy = (sensitivity + specificity) / 2; useful for binary classification with class imbalance [16].

Confidence intervals: Wilson score interval is recommended for binomial proportions, especially for small samples and extreme proportions (near 0 or 1); provides more stable results than normal approximation [18].

Effect size: Cohen's h for proportions (requires arcsine transformation); Cohen's d for continuous data [18].

Minimum sample size: Power analysis determines required N for detecting effect size α with given α/β; typically N > 200 per class for statistical significance [16].

Cross-validation: Stratified sampling maintains class distribution in train/test splits [16].

Calibration assessment: Calibration curves and ECE measure whether predicted confidence matches observed accuracy [3].

---

**PHASE 7: RECENT NEURO-SYMBOLIC WORK AND POSITIONING (2024-2025)**

IEEE Transactions on Pattern Analysis and Machine Intelligence (Feb 2025) published "Towards Data-And Knowledge-Driven AI: A Survey on Neuro-Symbolic Computing," systematizing neuro-symbolic research along neural-symbolic integration, knowledge representation, knowledge embedding, and functionality [2].

MDPI Mathematics (May 2025) published "AI Reasoning in Deep Learning Era: From Symbolic AI to Neural–Symbolic AI," providing comprehensive overview focusing on Neural–Symbolic AI's role as complement to LLMs, examining symbolic solvers for factual grounding [2].

Neural Computing and Applications (2024) survey: Neuro-symbolic AI develops human-like reasoning by combining symbolic reasoning with connectionist learning; addresses interpretability, explainability, and robust reasoning [2].

Key positioning insight: Demand-driven (lazy) evaluation reduces hallucination vs. eager translation (LINC, Logic-LM pattern) by extracting predicates only when the proof engine fails to derive the goal without them [4].

Confidence propagation mechanisms: Scallop reasons over probabilistic inputs via a provenance framework where AND is evaluated as multiplication, OR as minimization, NOT as 1-x [20]. Probabilistic Soft Logic (PSL) propagates labels via first-order constraints [20]. Neural LP learns rules with confidence scores [20].

Recent fusion patterns: (a) Selective symbolic translation with confidence-gating (HBLR) [4]; (b) Retrieval + symbolic reasoning (KG-RAG, hierarchical information retrieval RAPTOR) [21]; (c) Reflection modules for consistency checking [4]; (d) Solver feedback-guided rule generation (ARGOS) [12].

Emergence of "neurosymbolic faithful reasoning" emphasizing both accuracy and interpretable trace generation. Current state-of-the-art on ProofWriter ~80%, RuleTaker D5 varies by approach.

**White space for RFDE positioning**: Demand-driven extraction combining LLM translation (only on proof failure) + backward-chaining triggered fact extraction + Prolog verification with confidence propagation represents a distinct architectural contribution not fully explored by prior work [4, 14].

---

**PHASE 8: HONEST SCOPE DECLARATION PATTERNS**

ACL/EMNLP best practice from recent papers:

**Pattern 1: Explicit Problem Scoping**
Example from legal reasoning work [22]: "This approach addresses only a limited scope as an initial step—with experiments constrained in terms of problem space, architectural design, datasets, logic interpreters, prompt tuning, metrics, and LLMs. Currently, focus is only on health insurance coverage questions; application to civil and corporate legal terms remains out of scope."

**Pattern 2: Limitations as Design Trade-Offs**
Rather than framing constraints as failures, reframe as intentional choices: "We prioritize precision over coverage by requiring explicit rule specification" or "We optimize for single-hop reasoning chains over multi-hop to maintain interpretability and reduce hallucination."

**Pattern 3: Measurement Validity Acknowledgment**
Example: "We cannot determine whether accuracy gains derive from hallucination reduction or improved semantic parsing; we report both metrics independently to enable separate assessment."

**Pattern 4: Concrete Future Work**
Avoiding vague statements; instead: "Rule acquisition via ILP remains future work; this work assumes pre-specified Horn-clause rulesets. AMIE-style mining or LLM-guided synthesis (as in ARGOS) could address this in future iterations."

**Pattern 5: Negative Results and Per-Class Breakdown**
When a system underperforms: "System achieves 85% overall accuracy but only 40% on yes-label examples (N=200), indicating bias toward no-label defaults. We attribute this to class imbalance in training data and report per-class F1 for transparency."

**Pattern 6: Failure Analysis by Category**
Segment errors systematically: "Hallucination errors (22% of failures), parsing errors (15%), rule application errors (8%), with remaining 5% unexplained. Hallucination predominantly occurs in kinship relations requiring commonsense (e.g., in-law inference)."

**Standard Structure in ACL/EMNLP**: Scope statements appear in abstract (brief), introduction (motivation for scope), dedicated limitations section (detailed constraints), and future work (concrete next steps) [22].

---

**PHASE 9: CUSTOM CORPUS ANNOTATION AND EVALUATION**

Inter-annotator agreement (IAA) standard: Cohen's Kappa ≥0.75 indicates "substantial agreement" per Landis & Koch interpretation [23]. Kappa = (Po - Pe) / (1 - Pe), where Po = observed agreement, Pe = expected agreement by chance [23].

Predicate-level annotation schema: Based on PropBank conventions with verb-specific core roles (ARG0-ARG5, ARGM-TMP, ARGM-DIR, etc.) [13]. Each predicate should be annotated with:
- Predicate lemma (e.g., "have", "give", "increase")
- Arguments with roles (ARG0 = agent, ARG1 = patient, ARG2 = instrument, etc.)
- Span offsets pointing to source text [13]

Domain-specific annotation challenges:
- **Legal texts**: Domain expert review required; legal predicates and argument roles differ from general English (e.g., "indemnify", "tort") [23]
- **News articles**: Source-linked span annotation; claims must cite evidence [23]
- **Narrative texts (children's stories)**: Focus on entity relations and inferred commonsense rules (e.g., if X is Y's parent and Y is Z's parent, infer transitivity) [23]

Annotation guideline structure:
1. Definition of predicate types and roles
2. Worked examples per document type (legal, news, narrative)
3. Ambiguity resolution strategy (when predicate could have multiple roles)
4. Edge case handling (implicit arguments, coreference, dropped subjects) [23]

Minimum sample size: 50-document corpus (20 legal, 15 news, 15 narrative) yields ~1,000-2,000 predicate annotations; power analysis suggests N ≥ 50 per domain for robust IAA estimation [23].

Quality assurance procedure:
1. Dual annotation of 20% of documents; calculate per-item Kappa [23]
2. Adjudication meeting to resolve disagreements (Kappa gap >0.25) [23]
3. Iterative guideline refinement if IAA < 0.75 [23]
4. Final agreement calculation across full annotated set

Cross-domain evaluation: Assess whether rules learned on legal documents generalize to news; report per-domain accuracy and transfer metrics [23].

Bias mitigation:
- Randomly select documents; track annotator and document characteristics [23]
- Report annotator agreement breakdown (inter-rater Kappa by annotator pair)
- Identify systematic biases (e.g., one annotator over-predicting certain roles) [23]

Tooling: Prodigy (active learning annotation), BRAT (span annotation), or Label Studio (flexible UI) with JSON export for downstream processing [23].

---

**SYNTHESIS: KEY AUDIT FINDINGS FOR ITERATION 2**

1. **Measurement Separation**: Iteration 1 mechanically derived "0% hallucination" from "100% accuracy". Iteration 2 must measure atomic extraction hallucination independently via span grounding and FActScore-style fact decomposition [1, 2].

2. **Benchmark Enforcement**: Iteration 1 used hand-crafted tasks with 92% yes-labels. Iteration 2 must enforce published benchmarks (CLUTRR, RuleTaker D5, ProofWriter, FOLIO) with ≥40% non-yes labels and compositional evaluation splits [5, 6].

3. **Baseline Fairness**: Iteration 1 made no baseline comparisons. Iteration 2 must run LINC, Logic-LM, HBLR against same benchmarks, report per-class metrics, and use McNemar test for significance [4, 7, 8, 15].

4. **Rule Acquisition Scoping**: Iteration 2 must explicitly state that pre-specified Horn-clause rules are assumed; rule acquisition via ILP/mining is future work [17].

5. **Proof-Trace Audit**: Audit 20 test examples for task underutilization (direct goal query vs. multi-hop backward chaining); report depth distribution and percentage of tasks invoking ≥1 rule [14].

6. **Statistical Rigor**: Use McNemar test (p<0.05), per-class F1, macro-F1, 95% Wilson score CIs, Cohen's Kappa ≥0.75, N≥200 per class, stratified cross-validation [15, 16, 18, 23].

7. **Honest Positioning**: Situate RFDE in context of LINC's eager translation hallucination risk, LAMBADA's pre-extraction assumption, HBLR's parse-time confidence-gating, and ARGOS's solver-feedback-driven rule generation. Position demand-driven extraction as distinct contribution [4, 8, 11, 12].

8. **Scope Declaration**: Explicit scope section delineating what RFDE addresses (document extraction, logical reasoning, hallucination reduction via demand-driven extraction) and does NOT address (rule acquisition, multi-document reasoning, temporal reasoning).

9. **Publication Readiness**: Prepare annotated 50-document corpus (20 legal, 15 news, 15 narrative) with Cohen's Kappa ≥0.75 inter-annotator agreement; draft for ACL Knowledge Extraction track.

## Sources

[1] [How Much Do LLMs Hallucinate in Document Q&A Scenarios? A 172-Billion-Token Study](https://arxiv.org/html/2603.08274v1) — Defines hallucination as factuality (contradicting real information, fabricated) vs. faithfulness (deviating from instructions) violations; proposes FActScore for atomic fact decomposition; shows GPT-4 achieves only 13.4% precision on reference checking in systematic reviews.

[2] [Towards Data-And Knowledge-Driven AI: A Survey on Neuro-Symbolic Computing](https://www.computer.org/csdl/journal/tp/2025/02/10721277/2179549p9QY) — February 2025 IEEE survey systematizing neuro-symbolic computing along neural-symbolic integration, knowledge representation, knowledge embedding, and functionality dimensions; discusses LLM complement role.

[3] [Semantic Role Labeling (Chapter 21, SLP3)](https://web.stanford.edu/~jurafsky/slp3/21.pdf) — Foundational SRL overview; describes PropBank/FrameNet conventions for semantic roles; discusses calibration and confidence scoring in semantic extraction.

[4] [From Hypothesis to Premises: LLM-based Backward Logical Reasoning with Selective Symbolic Translation](https://arxiv.org/html/2512.03360) — HBLR framework (Li et al., 2026) with confidence-aware selective translation and reflection modules; demonstrates demand-driven extraction advantages; parse-time (not demand-driven) confidence-gating.

[5] [CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text](https://aclanthology.org/D19-1458/) — 9,074 training, 2,020 validation, 1,146 test examples; compositional testing on held-out rule combinations; distribution shifts from clause length generalization.

[6] [RuleTaker: A Dataset for Logical Reasoning](https://huggingface.co/datasets/jise/ruletaker) — 100,000 theories per depth (0-5); yes/no questions requiring 0-5 deductive steps; closed-world assumption; D5 is hardest variant.

[7] [LINC: A Neurosymbolic Approach for Logical Reasoning](https://aclanthology.org/2023.emnlp-main.313.pdf) — Combines StarCoder-15B with FOL prover; beats GPT-4 on ProofWriter by 10%; achieves ~75% precision with 26% improvement over CoT.

[8] [Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning](https://arxiv.org/pdf/2305.12295) — Maps NL to FOL/SAT for solver refinement; achieves 39.2% improvement over standard prompting, 18.4% over CoT; >20% hallucination from eager translation.

[9] [ProofWriter: A Dataset for Proof Generation](https://www.emergentmind.com/topics/proofwriter-dataset) — Synthetic NL logical reasoning with facts, rules, implications, proofs; supports entailment, proof generation, implication enumeration, abduction; open-world assumptions.

[10] [FOLIO: Natural Language Reasoning with First-Order Logic](https://arxiv.org/abs/2209.00840) — 1,435 examples, 487 premise sets; true/false/unknown labels; FOL-verified by inference engine; open-domain logical reasoning.

[11] [LAMBADA: Backward Chaining for Automated Reasoning in Natural Language](https://arxiv.org/abs/2212.13894) — Four-module backward chaining via few-shot prompting; outperforms forward reasoning; requires pre-extracted facts (cannot extract from documents themselves).

[12] [A Balanced Neuro-Symbolic Approach for Commonsense Abductive Logic](https://arxiv.org/html/2601.18595) — ARGOS balances linguistic and symbolic by guiding LLM-generated commonsense rules via SAT solver backbone; solver-failure-driven rule generation halves chain length.

[13] [The Proposition Bank: An Annotated Corpus of Semantic Roles](https://www.cs.rochester.edu/~gildea/palmer-propbank-cl.pdf) — Defines verb-specific semantic roles (ARG0-ARG5, ARGM modifiers); basis for predicate annotation schema; generalizes better to unseen predicates than FrameNet.

[14] [SLD-Resolution Proof Tree and Meta-Interpreter Approaches](https://www.researchgate.net/figure/SLD-Resolution-Proof-Tree_fig1_2382372) — Describes SLD resolution tree structure; proof obligation extraction; meta-interpreter approach for capturing derivation dependencies, variable bindings, and predicate classification.

[15] [McNemar's Test to Evaluate Machine Learning Classifiers with Python](https://towardsdatascience.com/mcnemars-test-to-evaluate-machine-learning-classifiers-with-python-9f26191e1a6b) — McNemar test for paired samples; 2×2 contingency table construction; chi-square statistic; standard for comparing classifier performance on matched pairs.

[16] [Mastering Classification Metrics: F1-Score to AUC-ROC](https://medium.com/@balaji92/mastering-classification-metrics-a-deep-dive-from-f1-score-to-auc-roc-87e82b0ebfae) — Per-class F1, macro-F1, micro-F1 definitions; balanced accuracy for imbalanced data; minimum sample sizes N≥200; stratified cross-validation importance.

[17] [Fast Rule Mining in Ontological Knowledge Bases with AMIE+](https://link.springer.com/article/10.1007/s00778-015-0394-1) — Rule mining via frequent pattern mining; ILP approaches; AMIE handles open-world assumption; scales 3 orders of magnitude faster than prior systems.

[18] [The Wilson Confidence Interval for a Proportion](https://www.econometrics.blog/post/the-wilson-confidence-interval-for-a-proportion/) — Wilson score interval for binomial proportions; outperforms normal approximation for small N and extreme proportions; stable 95% CI calculation.

[19] [YAGO 4.5: A Large and Clean Knowledge Base with a Rich Taxonomy](https://arxiv.org/pdf/2308.11884) — YAGO integrates Wikipedia, WordNet, SUMO upper ontology; OpenCyc lacks axioms; DBpedia less consistent; comparison of upper ontologies for rule grounding.

[20] [Towards Probabilistic Inductive Logic Programming with Neurosymbolic Inference and Relaxation](https://arxiv.org/pdf/2408.11367) — Confidence propagation via provenance framework (AND=product, OR=min, NOT=1-x); Scallop probabilistic reasoning; neural LP with confidence scores.

[21] [StructRAG: Boosting Knowledge Intensive Reasoning via Hybrid Information Structurization](https://arxiv.org/pdf/2410.08815) — Retrieval + symbolic reasoning fusion; hierarchical information retrieval (RAPTOR); KG-RAG integration for multi-hop reasoning improvement.

[22] [Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law](https://arxiv.org/pdf/2502.17638) — Example of honest scope declaration: explicitly lists constrained problem space, datasets, logic interpreters, metrics, and LLMs; delineates out-of-scope applications.

[23] [Inter-Annotator Agreement: An Introduction to Cohen's Kappa Statistic](https://surge-ai.medium.com/inter-annotator-agreement-an-introduction-to-cohens-kappa-statistic-dcc15ffa5ac4) — Cohen's Kappa ≥0.75 as substantial agreement standard; annotation guidelines structure; dual annotation, adjudication, iterative refinement; domain-specific challenges.

## Follow-up Questions

- What specific per-class F1 scores did LINC and Logic-LM achieve on CLUTRR and RuleTaker when broken down by label (yes/no/unknown)? Can we obtain their published confusion matrices to assess fairness across classes?
- How does demand-driven (lazy) extraction compare to eager translation in terms of actual hallucination rate (measured via span grounding) on a standardized benchmark? A controlled experiment comparing both approaches on the same document set and ruleset would validate the architectural assumption.
- For the 50-document custom corpus, what is the inter-rater agreement breakdown by document type (legal vs. news vs. narrative)? Do certain domains show systematic biases in predicate annotation that should inform domain-stratified evaluation on benchmarks?

---
*Generated by AI Inventor Pipeline*
