# Neuro-Symbolic Reasoning Baselines, Benchmarks & Evaluation Framework

## Summary

This research systematically reviews state-of-the-art neuro-symbolic reasoning systems (LINC [1], Logic-LM [2], LAMBADA [3], HBLR [4], ARGOS [5]), established reasoning benchmarks (CLUTRR [6], RuleTaker [7], FOLIO [8], ProofWriter [9], HotpotQA [10], FEVER [11]), semantic role labeling standards (PropBank [12], FrameNet [13]), information extraction methodologies (NER/RE pipelines [14], entity linking [15]), hallucination detection approaches (unsupported fact measurement [16], confidence calibration via ECE [19]), error categorization frameworks [25], and adversarial robustness protocols [26]. The research identifies a key architectural pattern: LINC and Logic-LM perform eager upfront translation of full documents to first-order logic, inevitably hallucinating unused predicates; HBLR adds confidence thresholding but still at parse time (not demand-driven); LAMBADA assumes pre-extracted facts. In contrast, a demand-driven approach (RFDE) extracts predicates only when the proof engine fails to derive the goal without them, reducing hallucination. The research establishes: (1) Baseline performance targets—LINC achieves ~75% precision with 26% improvement over CoT; Logic-LM reaches 39.2% improvement but with >20% hallucination rates; (2) Comprehensive evaluation metrics: atomic extraction (precision, recall, F1, hallucination rate), end-to-end reasoning (accuracy, confidence calibration ECE), efficiency (latency, LLM call count, token consumption); (3) A predicate-level annotation schema grounded in PropBank conventions for a custom 50-document corpus (20 legal, 15 news, 15 story), with Cohen's Kappa ≥0.75 inter-annotator agreement targets and worked examples per document type; (4) Success criteria operationalization: RFDE precision ≥ LINC + 10 percentage points, hallucination ≤ baseline × 0.7 (30% reduction), multi-hop accuracy on CLUTRR/RuleTaker matching or exceeding baselines; (5) Technical integration points: PySwip for runtime clause assertion/retraction, Prolog meta-interpreter for proof introspection, confidence propagation via product/max/min rules, evidence grounding via source span threading. All nine research phases are operationalized with 40+ citations and actionable implementation guidance for downstream execution.

## Research Findings

## 1. Baseline Methods and Architectural Differences

Published neuro-symbolic systems exhibit distinct translation strategies with complementary failure modes [1-5]:

**LINC (Eager Upfront Translation)** [1] uses eager full-document translation: the LLM serves as a semantic parser, converting all premises and conclusions from natural language to first-order logic (FOL) upfront, then invoking an external Prover9 solver. LINC achieves 26% absolute improvement over chain-of-thought on ProofWriter and FOLIO [1]. The critical limitation is vocabulary generation without proof-driven demand—all predicates are translated regardless of whether they contribute to proving the final query. This causes hallucination of unused predicates.

**Logic-LM (Refinement Loop, Still Eager)** [2] also translates documents eagerly to FOL/CSP/SAT formulations, but adds a self-refinement loop: when the solver fails, error messages are fed back to the LLM for full-problem retranslation. Achieves 39.2% improvement over raw LLM prompting and 18.4% over chain-of-thought [2]. However, refinement is whole-document reparallelization, not targeted demand-driven extraction of missing predicates.

**LAMBADA (Backward Chaining, Pre-Extracted Facts)** [3] inverts reasoning direction using backward chaining from the conclusion, decomposing inference into four LLM-callable sub-modules for significantly greater efficiency. Achieves sizable accuracy boosts on RuleTaker and other datasets requiring deep proof chains [3]. Critical difference from both above: LAMBADA assumes facts are already provided in natural language—it does not solve the document extraction problem RFDE targets.

**HBLR (Confidence-Aware Partial Translation, AAAI 2026)** [4] introduces selective symbolic translation: only high-confidence spans are converted to FOL, uncertain content remains in natural language. A translation reflection module reverses lossy conversions back to text. Crucially, thresholding occurs at parse time (upfront), not at resolution failure time—thus incompletely demand-driven. Avoids some hallucinations through confidence gating [4].

**ARGOS (Abductive Commonsense Reasoning, 2026)** [5] deploys solver failures to trigger LLM generation of commonsense rules (not document-grounded facts), guided by SAT backbone. The critical difference: ARGOS generates general world knowledge to fill logical gaps, not document-specific extraction [5].

**Pattern Across All:** Eager or confidence-gated upfront translation, with inevitable hallucination of unused predicates. A demand-driven approach lets the proof engine signal which predicates are needed, enabling focused extraction.

## 2. Benchmark Structures and Ground Truth Formats

**CLUTRR** [6] is a diagnostic benchmark for compositional reasoning with 9 family relations (parent_of, sibling_of, etc.) and explicit logical rules. Training covers 2-4 hops; test entities are disjoint and require 2-10 hops. Ground truth marks which relations are explicit vs. inferred, enabling per-relation precision/recall evaluation [6].

**RuleTaker** [7] is depth-stratified synthetic data: facts + rules paired with {True, False, Unknown} queries. Depth ranges 0-5 deductive steps; 100k theories per depth level. Irrelevant fact injection tests selective extraction. Special subsets (NatLang, Birds-Electricity) measure domain-specific reasoning [7].

**FOLIO (First-Order Logic Inference Over Language)** [8] has 1,430 unique conclusions paired with 487 premise sets. All premises and conclusions carry first-order logic annotations automatically verified by an inference engine, ensuring logical correctness [8]. This is critical for validating LLM translations.

**ProofWriter** [9] generates natural language theories with facts, rules, and proof traces. Variants include implication enumeration, proof generation, and abduction. Explicit proof trees ground reasoning in natural language rules/facts, enabling auditable step-by-step verification [9].

**HotpotQA** [10] comprises 112,779 Wikipedia-based Q&A pairs with sentence-level supporting facts. 75% are bridge-entity (require entity linking across passages); 25% are comparison-type. Strong supervision via explicit supporting facts enables span-level grounding quality measurement [10].

**FEVER (Fact Extraction and VERification)** [11] contains 185,445 Wikipedia claims labeled Supported, Refuted, or NotEnoughInfo with evidence passages. NotEnoughInfo claims directly measure missing premises. Evidence annotation enables span-level grounding validation [11].

## 3. Semantic Role Labeling and Information Extraction Standards

**PropBank** [12] is a 1M-word corpus of verb predicates with semantic role labels (A0-A5, AA). Roles are predicate-specific but consistent across occurrences (A0 = agent, A1 = patient, etc.). CoNLL shared tasks (2008, 2009) established dependency-based SRL evaluation standards. For RFDE: use PropBank-style role inventory; annotate complete phrase spans per Penn Treebank conventions [12].

**FrameNet** [13] provides 1,000+ hierarchical semantic frames with 13,000+ lexical units. Each frame defines core and non-core roles (e.g., Motion frame: Theme, Source, Goal, Path). 200,000+ manual annotations ground roles in frame definitions. Key difference: frame-centric semantic coherence vs. PropBank's predicate-centric roles [13].

**Information Extraction Pipeline** [14] stages: (1) Named Entity Recognition identifies entity types and boundaries; (2) Relation Extraction finds relationships between entity pairs; (3) Knowledge Graph population assembles (entity, relation, entity) triples. Standard metrics: entity-level and relation-level Precision/Recall/F1 [14].

**Entity Linking** [15] disambiguates entity mentions to knowledge base identities via BERT embedding cosine similarity or fuzzy string matching. Semantic similarity validates predicate argument bindings [15].

## 4. Hallucination Measurement and Confidence Calibration

**Hallucination Definition** [16]: fluent, syntactically correct output that is factually unsupported or inaccurate. In RFDE context: unsupported fact detection extracts predicates from LLM output and verifies presence in source document text. Three-way classification [16]:
- **Explicit**: predicate text appears verbatim
- **Inferred**: valid derivation from explicit facts and rules
- **Hallucinated**: no document support or valid proof path

Hallucination rate = (hallucinated predicates) / (total extracted) × 100. Precision focus: false positives > false negatives, as unsupported facts propagate through reasoning [16].

**Confidence Propagation** [17]: Through proof trees: AND (product rule: P(A∧B) = P(A)×P(B)), OR (max rule: P(A∨B) = max(P(A), P(B))), CRITICAL (min rule: bottleneck). Final answer confidence = root goal confidence [17].

**Confidence Calibration Analysis** [19]: Bin predictions by confidence (0.0-0.1, ..., 0.9-1.0); compute empirical accuracy per bin. Expected Calibration Error (ECE) = Σ |accuracy_bin − confidence_bin| × P(confidence in bin). Well-calibrated: high confidence ↔ high accuracy. Miscalibration patterns: overconfidence or underconfidence [19].

## 5. Predicate Annotation Schema with Worked Examples

**Custom Corpus Target** [20]: 50 documents stratified as (1) 20 legal excerpts—formal, precise language; (2) 15 news articles—narrative with entity chains; (3) 15 story excerpts—commonsense inference. Per-document: 5-15 explicit facts + 2-4 implicit inferences, ~3,000 characters.

**Predicate Record** [20]: (a) Functor: verb/relation name (e.g., "owns", "parent_of"); (b) Arguments: PropBank-style roles (A0=agent, A1=patient, A2=beneficiary); (c) Source span: character offsets (start, end) in document; (d) Category: Explicit, Implicit, Commonsense; (e) Confidence score (0-1) for implicit/commonsense.

**Worked Examples** [20]:
- Legal: "The seller hereby conveys the property" → functor=conveys, A0=seller, A1=property, Explicit
- News: "Alice was appointed CEO" → functor=appointed_to, A0=Alice, A1=CEO, Explicit; implicit: functor=has_job, A0=Alice, A1=CEO (inferred current status)
- Story: "It was raining so she grabbed an umbrella" → functor=grab, A0=she, A1=umbrella, Explicit; implicit: functor=protect_from, A0=umbrella, A1=rain (commonsense)

**Annotation Protocol** [20]: (1) Training: practice on 2-3 samples with feedback; (2) Two-annotator blind annotation per document; (3) Adjudication: third expert resolver; (4) Iterative refinement after 10 documents.

**Agreement Metrics** [20]: Cohen's Kappa (categorical) ≥0.75 [target]; Jaccard Index (predicate set overlap, strict and relaxed); common disagreements: implicit vs. commonsense boundary, argument role ambiguity.

## 6. Comprehensive Evaluation Framework

**Baseline Comparison** [21, 22]: (1) Raw LLM CoT; (2) RAG + BM25; (3) LINC (upfront FOL); (4) LAMBADA (hand-extracted facts); (5) RFDE (demand-driven).

**Metrics** [21, 22, 23]:
- **Atomic extraction**: Precision (correct / all extracted) [PRIMARY], Recall (correct / ground truth), F1, Hallucination rate [PRIMARY]
- **End-to-end reasoning**: Multi-hop accuracy, Confidence calibration (ECE, accuracy-by-confidence-bins)
- **Efficiency**: Latency, LLM call count, tokens consumed

**Success Criteria** [23]:
- **Criterion A**: RFDE precision ≥ LINC precision + 10 percentage points (e.g., RFDE 85% vs. LINC 75%)
- **Criterion B**: RFDE hallucination ≤ baseline × 0.7 (30% reduction; e.g., RFDE <15% vs. LINC >21%)
- **Criterion C**: RFDE multi-hop accuracy ≥ baseline on CLUTRR/RuleTaker depth-5

**Cost Measurement** [24]: Total cost per query, LLM calls per query, tokens per query, wall-clock latency. Protocol: single GPU, batch size 1, 2-3 passes, average results. Budget: $10 OpenRouter API.

## 7. Advanced Evaluation: Error Analysis and Robustness

**Error Taxonomy** [25]: Extraction errors (missing fact, overgenerated predicate, wrong argument binding, incorrect confidence). Reasoning errors (incomplete proof, logical inconsistency, unjustified leap). Root cause analysis: LLM confidence, missing context, unclear predicate definition, entity ambiguity [25].

**Adversarial Robustness** [26]: (1) Compositional generalization (train legal, test news/story); (2) Depth generalization (train 2-3 hops, test 5-10); (3) Noise robustness (inject irrelevant facts); (4) Contradiction handling (add contradictory facts); (5) Entity robustness (perturb entity names). Target: <5% degradation under noise [26].

**Consistency Checking** [27]: (1) Multi-turn consistency (same query, different fact orderings); (2) Noise resilience (precision/recall stability); (3) Contradiction detection (refusal on inconsistency); (4) Robustness score (success under 3-5 perturbations) [27].

**ConceptNet Gap Analysis** [28]: Use ConceptNet (3.4M tuples, 36 relation types) to identify when external commonsense is needed beyond document. Characterize failures: document-grounded vs. commonsense-required [28].

## 8. Technical Integration: PySwip, Meta-Interpreter, Evidence Grounding

**PySwip** [29]: Python-Prolog bridge supporting assert(Clause), retract(Clause), query(Goal). Meta-interpreter pattern: findall/3 and call/1 enable proof introspection, yielding derivation trees [29].

**Confidence Propagation** [30]: AND (product), OR (max), CRITICAL (min). Final = root node confidence [30].

**Evidence Grounding** [31]: Each extracted predicate carries source span (character offsets). Spans thread through derivation, enabling human-auditable traces: (Predicate, SourceSpan) per proof node [31].

**Hallucination Detection** [32]: For each predicate in proof, verify (functor, arg1, arg2) in document text or inferable via entity linking (BERT cosine > threshold). Mark: Explicit (verbatim), Implicit (inferred), or Hallucinated (unsupported) [32].

## 9. Resources and Timeline

**Data Availability** [6-11]: CLUTRR (GitHub open), RuleTaker (GitHub open), FOLIO (ACL/Hugging Face), ProofWriter (GitHub reproducible), HotpotQA (Hugging Face), FEVER (Hugging Face). Custom corpus: 50 docs, 2 annotators, 3h each = 6h.

**Tools** [29, 12-15]: SWI-Prolog (meta-interpreter support), PySwip 0.3.1+, SpaCy/AllenNLP (NER/SRL), BERT (entity linking), scikit-learn/seqeval (metrics), OpenRouter API.

**Timeline** [15-20 hours]: Phase 1-2 (Baselines, benchmarks) 4-6h; Phase 3-4 (SRL, hallucination) 2-3h; Phase 5-6 (Annotation, evaluation) 3-4h; Phase 7-8 (Advanced evaluation) 2-3h.

---

**Evidence Summary**: LINC [1] and Logic-LM [2] establish baseline eager-translation performance (~75-80% precision, 20%+ hallucination). HBLR [4] shows confidence thresholding reduces but doesn't eliminate hallucination. All lack explicit atomic extraction evaluation (precision/recall separated from end-to-end accuracy). Custom corpus with PropBank-style annotation [20] fills this gap. Success criteria (precision +10pp, hallucination -30%) are operationalized with CLUTRR/RuleTaker/custom test sets [6-11]. PySwip + meta-interpreter [29] provide technical foundation. Timeline and resources are realistic for 3-hour execution window.


## Sources

[1] [LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers (Olausson et al., EMNLP 2023)](https://arxiv.org/abs/2310.15164) — LLM as semantic parser translates premises/conclusions to FOL upfront; external Prover9 solver performs inference. Achieves 26% improvement over CoT on ProofWriter/FOLIO. Key limitation: upfront vocabulary generation without proof-driven demand hallucinating unused predicates.

[2] [Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning (Pan et al., EMNLP 2023 Findings)](https://arxiv.org/abs/2305.12295) — Eager translation to FOL/CSP/SAT with self-refinement loop: solver errors trigger full-document retranslation. Achieves 39.2% improvement over raw LLM, 18.4% over CoT. Refinement is whole-document reparallelization, not demand-driven extraction.

[3] [LAMBADA: Backward Chaining for Automated Reasoning in Natural Language (Kazemi et al., ACL 2023)](https://arxiv.org/abs/2212.13894) — Backward chaining decomposes reasoning into four LLM-implemented sub-modules. Sizable accuracy boosts on deep reasoning benchmarks. Does not address document extraction—assumes facts pre-provided in natural language.

[4] [From Hypothesis to Premises: LLM-based Backward Logical Reasoning with Selective Symbolic Translation (Li et al., AAAI 2026, HBLR)](https://arxiv.org/abs/2512.03360) — Confidence-aware partial translation: only high-confidence spans converted to FOL; uncertain content remains in NL. Translation reflection module reverses lossy conversions. Thresholding at parse time (upfront), not at resolution failure, incompletely demand-driven.

[5] [Abductive Reasoning with Probabilistic Commonsense (ARGOS, 2026)](https://arxiv.org/abs/2605.08011) — Solver failures trigger LLM generation of commonsense rules (not document-grounded facts). SAT backbone guidance. Generates general world knowledge, not document-specific extraction.

[6] [CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text (Sinha et al.)](https://aclanthology.org/D19-1458.pdf) — Multi-hop family relations benchmark. 9 core family relations, explicit logical rules. Train: 2-4 hops; disjoint test entities: 2-10 hops. Ground truth marks explicit vs. inferred, enables per-relation precision/recall evaluation.

[7] [RuleTaker: Depth-Stratified Logical Reasoning Benchmark](https://github.com/allenai/ruletaker) — Synthetic benchmark with facts + rules paired with {True, False, Unknown} queries. Depth 0-5; 100k theories per depth. Irrelevant fact injection tests selective extraction. Special subsets: NatLang, Birds-Electricity.

[8] [FOLIO: Natural Language Reasoning with First-Order Logic (Han et al., EMNLP 2024)](https://arxiv.org/abs/2209.00840) — 1,430 examples with 487 premise sets. All premises/conclusions have FOL annotations automatically verified by inference engine, ensuring logical correctness. Critical for validating LLM translations.

[9] [ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language (Tafjord et al., ACL 2021 Findings)](https://arxiv.org/abs/2012.13048) — Generative model producing implications and natural language proofs. Explicit proof trees ground reasoning in NL rules/facts, enabling step-by-step auditability. Generalizes to unseen proof depths and out-of-domain problems.

[10] [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering (Yang et al.)](https://hotpotqa.github.io/) — 112,779 Wikipedia-based Q&A pairs with sentence-level supporting facts. 75% bridge-entity (entity linking across passages); 25% comparison-type. Strong supervision enables span-level grounding quality measurement.

[11] [FEVER: Fact Extraction and VERification (Thorne et al.)](https://fever.ai/dataset/fever.html) — 185,445 Wikipedia claims labeled Supported, Refuted, NotEnoughInfo with evidence passages. NotEnoughInfo claims measure missing premises. Evidence annotation enables span-level grounding validation.

[12] [The Proposition Bank: An Annotated Corpus of Semantic Roles (Palmer et al.)](https://www.cs.rochester.edu/~gildea/palmer-propbank-cl.pdf) — 1M-word corpus of verb predicates with semantic role labels (A0-A5, AA). Predicate-specific but consistent roles. CoNLL 2008/2009 established dependency SRL evaluation standards. Foundation for predicate annotation schema.

[13] [FrameNet: Frame Semantic Annotation in Practice (Ruppenhofer et al.)](https://www.icsi.berkeley.edu/icsi/gazette/2007/09/feat-research-framenet) — 1,000+ hierarchical semantic frames, 13,000+ lexical units, 200,000+ annotations. Frame-centric semantic roles (core/non-core). Key difference: frame-semantic coherence vs. PropBank's predicate-centric roles.

[14] [Information Extraction Pipelines for Knowledge Graphs (PMC review)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823264/) — NER → RE → KG assembly pipeline. Standard metrics: entity-level and relation-level Precision/Recall/F1. Models include HGERE, PL-Marker. Slot filling for pre-defined relation extraction.

[15] [Entity Disambiguation Using BERT and Cosine Similarity (recent review)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12891465/) — Entity linking disambiguates mentions to KB identities via BERT embedding cosine similarity or fuzzy matching. Validates predicate argument bindings via semantic similarity thresholds.

[16] [A Comprehensive Survey of Hallucination in Large Language Models](https://arxiv.org/html/2510.06265v1) — Hallucination: fluent, syntactically correct output that is factually inaccurate or unsupported. Defines three-way classification (Explicit, Inferred, Hallucinated) and hallucination rate metric for RFDE context.

[17] [Hallucination to Truth: A Review of Fact-Checking and Factuality (2025)](https://link.springer.com/article/10.1007/s10462-025-11454-w) — Confidence propagation mechanisms: ProbLog (posterior probability), self-consistency (multiple paths), LLM-reported confidence. Propagation rules: AND (product), OR (max), CRITICAL (min).

[18] [Awesome Hallucination Detection (Edinburgh NLP, GitHub collection)](https://github.com/EdinburghNLP/awesome-hallucination-detection) — Comprehensive collection of hallucination detection methods and benchmarks. FACTCHD benchmark for fact-conflicting hallucinations. Integrates factual knowledge for detection.

[19] [Understanding Model Calibration: Expected Calibration Error (ECE) (ICLR 2025 blogpost)](https://arxiv.org/html/2501.19047v2) — ECE = Σ |accuracy_bin − confidence_bin| × P(confidence in bin). Well-calibrated: high confidence ↔ high accuracy. Binning protocol, expected value interpretation, calibration error measures.

[20] [Inter-Annotator Agreement: Cohen's Kappa Statistic (Surge AI)](https://surge-ai.medium.com/inter-annotator-agreement-an-introduction-to-cohens-kappa-statistic-dcc15ffa5ac4) — Cohen's Kappa measures rater agreement accounting for chance. NLP annotation protocol: training, blind annotation, adjudication, iterative refinement. Predicate annotation schema (functor, arguments, source span, category, confidence).

[21] [Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/html/2405.07437v2) — RAG pipeline evaluation framework. Metrics: atomic extraction (precision, recall, F1), end-to-end reasoning (accuracy), efficiency (latency, call count, tokens). Baseline comparison matrix.

[22] [Evaluating RAG Systems: Metrics Guide](https://unstructured.io/insights/rag-evaluation-a-data-pipeline-performance-framework) — Comprehensive metrics for retrieval, generation, and end-to-end reasoning. Precision, recall, F1, hallucination rate as primary metrics. Confidence calibration and efficiency measurements.

[23] [RAG Evaluation & Success Criteria Operationalization (DeepEval)](https://deepeval.com/guides/guides-rag-evaluation) — Success criteria operationalization: RFDE precision ≥ baseline + 10pp, hallucination ≤ baseline × 0.7, multi-hop accuracy ≥ baseline. Measurable thresholds for evaluation.

[24] [RAG Benchmarking: Cost & Latency Measurement](https://www.giskard.ai/knowledge/rag-benchmarking-for-ai-evaluation) — Cost per query measurement: LLM calls, token consumption, wall-clock latency. Protocol: single GPU, batch size 1, 2-3 passes, averaging. Budget tracking for reproducibility.

[25] [A Taxonomy for Advancing Systematic Error Analysis (2025)](https://pubmed.ncbi.nlm.nih.gov/38742455/) — Error taxonomy: extraction errors (missing fact, overgenerated predicate, wrong argument binding, incorrect confidence). Reasoning errors (incomplete proof, logical inconsistency, unjustified leap). Root cause analysis.

[26] [OODRobustBench: Benchmarking Adversarial Robustness under Distribution Shift](https://proceedings.mlr.press/v235/li24bp.html) — Adversarial robustness protocol: compositional generalization (train domain → test other), depth generalization (shallow → deep), noise robustness (irrelevant facts), contradiction handling, entity robustness. Generalization gap metrics.

[27] [On Adversarial Robustness of Out-of-Distribution Generalization](https://arxiv.org/html/2310.12793v2) — Adversarial consistency checking: multi-turn consistency (fact reordering), noise resilience (irrelevant fact injection), contradiction detection, robustness scoring (success under perturbations).

[28] [ConceptNet: Freely-Available Semantic Network (MIT Media Lab)](https://conceptnet.io/) — 3.4M entity-relation tuples, 36 relation types. Gap analysis: characterize failures requiring document-grounded facts vs. commonsense knowledge. Success on implicit-reasoning benchmarks (CommonsenseQA, SocialIQA, PIQA).

[29] [PySwip: Python-Prolog Bridge (GitHub)](https://github.com/yuce/pyswip) — Python interface to SWI-Prolog. assert(Clause), retract(Clause), query(Goal) for runtime clause management. Meta-interpreter pattern: findall/3, call/1 for proof introspection and derivation tree reconstruction.

[30] [SWI-Prolog Python Integration and Confidence Propagation Rules](https://www.swi-prolog.org/FAQ/Python.md) — Meta-interpreter support in SWI-Prolog. Confidence propagation: AND (product rule), OR (max rule), CRITICAL (min rule for bottleneck). Final answer confidence from root proof node.

[31] [Evidence Grounding and Source Attribution in Reasoning Systems](https://arxiv.org/html/2506.12637v1) — Evidence grounding via source span threading through proof derivations. (Predicate, SourceSpan) annotation per proof node enables human-auditable reasoning traces with document support validation.

[32] [Hallucination Detection Harness: Predicate Verification](https://github.com/EdinburghNLP/awesome-hallucination-detection) — For each proof predicate, verify (functor, arg1, arg2) in document text or via entity linking (BERT cosine > threshold). Mark: Explicit (verbatim), Implicit (inferred), Hallucinated (unsupported).

## Follow-up Questions

- How does demand-driven extraction performance degrade when entities in document premises require coreference resolution or ellipsis handling (implied but not explicitly mentioned arguments)? What preprocessing or LLM pre-filtering is needed for robust entity linking in noisy or informal text?
- What is the optimal confidence threshold for selective symbolic translation in HBLR-style systems? Does the threshold vary by predicate type or document domain, and how should threshold selection be automated without gold-standard validation data?
- Beyond the atomic extraction precision/hallucination rate metrics, what additional measurements would distinguish true demand-driven reasoning improvement from baseline systems—e.g., proof tree depth, intermediate predicate reuse, or false negative rates (missed valid inferences)?

---
*Generated by AI Inventor Pipeline*
