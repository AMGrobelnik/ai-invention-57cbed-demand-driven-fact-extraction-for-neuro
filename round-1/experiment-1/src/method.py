#!/usr/bin/env python3
"""
Resolution-Failure-Directed Extraction (RFDE):
A neuro-symbolic pipeline that uses backward-chaining SLD resolution failures
as on-demand triggers for LLM-based fact extraction from natural-language documents.

Compares against three baselines: Chain-of-Thought, BM25-RAG, and LINC-style
eager FOL translation.
"""

import gc
import json
import math
import os
import re
import resource
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from loguru import logger
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── Logging ────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ─── Hardware / Memory budget ────────────────────────────────────────────────
_avail = 22 * 1024**3  # ~22 GB available (detected above)
RAM_BUDGET = 4 * 1024**3  # 4 GB is plenty for this text-only workload
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ─── Config ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Budget tracking
COST_TRACKER: dict[str, float] = {"total": 0.0, "calls": 0}
HARD_BUDGET = 9.50  # stop at $9.50, hard limit $10

# Model selection (cheap for grounding, slightly smarter for baselines)
GROUNDING_MODEL = "meta-llama/llama-3.1-8b-instruct"   # reliable text output
BASELINE_MODEL = "meta-llama/llama-3.1-8b-instruct"    # same model for fair comparison

# Cost estimates per 1k tokens (input + output combined)
MODEL_COST_PER_1K = {
    "meta-llama/llama-3.2-3b-instruct": 0.00006,
    "meta-llama/llama-3.1-8b-instruct": 0.00010,
}


# ─── OpenRouter LLM Client ───────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    prompt: str,
    model: str = GROUNDING_MODEL,
    system: str = "You are a precise logical reasoning assistant.",
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> str:
    """Call OpenRouter LLM, track cost, enforce budget."""
    if COST_TRACKER["total"] >= HARD_BUDGET:
        raise RuntimeError(f"Budget exhausted: ${COST_TRACKER['total']:.4f}")

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"].get("content")
    if content is None:
        # Model returned tool_call or empty content — treat as unknown
        raise ValueError(f"LLM returned null content (finish_reason={data['choices'][0].get('finish_reason')})")
    text = content.strip()
    usage = data.get("usage", {})
    total_tokens = usage.get("total_tokens", len(prompt.split()) + len(text.split()))
    cost = (total_tokens / 1000) * MODEL_COST_PER_1K.get(model, 0.0002)

    COST_TRACKER["total"] += cost
    COST_TRACKER["calls"] += 1

    logger.debug(
        f"LLM call #{COST_TRACKER['calls']} | model={model} | tokens={total_tokens} | "
        f"cost=${cost:.6f} | cumulative=${COST_TRACKER['total']:.4f} | t={time.time()-t0:.2f}s"
    )
    logger.debug(f"  prompt[:200]={prompt[:200]!r}")
    logger.debug(f"  response[:200]={text[:200]!r}")
    return text


# ─── RFDE: Pure-Python SLD Resolution Engine ─────────────────────────────────

@dataclass
class Term:
    """A first-order logic term: either a constant (str) or variable (prefixed ?)."""
    value: str

    @property
    def is_var(self) -> bool:
        return self.value.startswith("?")

    def __repr__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Term) and self.value == other.value


@dataclass
class Atom:
    """A ground or partially-ground predicate: pred(arg1, arg2, ...)."""
    pred: str
    args: list[Term]

    def __repr__(self) -> str:
        return f"{self.pred}({', '.join(str(a) for a in self.args)})"

    def arity(self) -> int:
        return len(self.args)


@dataclass
class Clause:
    """A Horn clause: head :- body (body empty = fact)."""
    head: Atom
    body: list[Atom] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.body:
            return f"{self.head} :- {', '.join(str(b) for b in self.body)}"
        return str(self.head)


def _var(name: str) -> Term:
    return Term(f"?{name}")


def _const(v: str) -> Term:
    return Term(v.lower().strip().replace(" ", "_"))


def _atom(pred: str, *args: str) -> Atom:
    return Atom(pred, [_const(a) for a in args])


Substitution = dict[str, Term]


def apply_sub(atom: Atom, sub: Substitution) -> Atom:
    """Apply substitution to an atom."""
    new_args = []
    for a in atom.args:
        if a.is_var and a.value[1:] in sub:
            new_args.append(sub[a.value[1:]])
        else:
            new_args.append(a)
    return Atom(atom.pred, new_args)


def unify(a: Atom, b: Atom, sub: Substitution) -> Optional[Substitution]:
    """Try to unify two atoms; return extended substitution or None."""
    if a.pred != b.pred or len(a.args) != len(b.args):
        return None
    result = dict(sub)
    for ta, tb in zip(a.args, b.args):
        # Walk ta
        while ta.is_var and ta.value[1:] in result:
            ta = result[ta.value[1:]]
        while tb.is_var and tb.value[1:] in result:
            tb = result[tb.value[1:]]
        if ta == tb:
            continue
        if ta.is_var:
            result[ta.value[1:]] = tb
        elif tb.is_var:
            result[tb.value[1:]] = ta
        else:
            return None
    return result


def freshen(clause: Clause, counter: list[int]) -> Clause:
    """Rename variables in a clause to avoid collisions."""
    n = counter[0]
    counter[0] += 1
    mapping: dict[str, str] = {}

    def rename_term(t: Term) -> Term:
        if not t.is_var:
            return t
        vname = t.value[1:]
        if vname not in mapping:
            mapping[vname] = f"{vname}_{n}"
        return Term(f"?{mapping[vname]}")

    def rename_atom(a: Atom) -> Atom:
        return Atom(a.pred, [rename_term(t) for t in a.args])

    return Clause(rename_atom(clause.head), [rename_atom(b) for b in clause.body])


@dataclass
class ProofNode:
    goal: str
    source: str  # "kb", "llm", "rule", "failed"
    confidence: float
    children: list["ProofNode"] = field(default_factory=list)
    llm_response: str = ""


class RFDEEngine:
    """
    Resolution-Failure-Directed Extraction engine.

    On any unresolvable goal, calls LLM with the source document as context
    to ground the missing predicate, then asserts it and retries resolution.
    """

    def __init__(self, document: str, max_depth: int = 10, max_llm_calls: int = 20):
        self.document = document
        self.max_depth = max_depth
        self.max_llm_calls = max_llm_calls
        self.kb: list[Clause] = []
        self.llm_calls_used = 0
        self._counter = [0]
        self._llm_cache: dict[str, tuple[str, float]] = {}  # memoize LLM calls
        self.proof_trace: list[ProofNode] = []

    def assert_fact(self, pred: str, *args: str, confidence: float = 1.0) -> None:
        clause = Clause(head=_atom(pred, *args))
        clause.head.args.append(Term(f"_conf_{confidence:.3f}"))  # confidence tag
        # Remove confidence tag — store separately
        clause = Clause(head=_atom(pred, *args))
        self.kb.append(clause)

    def assert_rule(self, head: Atom, body: list[Atom]) -> None:
        self.kb.append(Clause(head=head, body=body))

    def _ground_with_llm(self, goal: Atom) -> tuple[str, float]:
        """Ask LLM whether goal holds given the document. Returns (yes/no/unknown, confidence)."""
        key = repr(goal)
        if key in self._llm_cache:
            return self._llm_cache[key]

        if self.llm_calls_used >= self.max_llm_calls:
            return "unknown", 0.0

        args_str = ", ".join(str(a) for a in goal.args)
        readable_args = " and ".join(
            a.value.replace("_", " ") for a in goal.args if not a.is_var
        )
        pred_readable = goal.pred.replace("_", " ")

        prompt = (
            f"Document: \"{self.document}\"\n\n"
            f"Based only on the document above, does the following statement hold?\n"
            f"Statement: {pred_readable}({readable_args})\n\n"
            f"Reply on a single line as: ANSWER: yes/no/unknown | CONFIDENCE: 0.0-1.0 | EVIDENCE: <brief quote>"
        )

        try:
            text = call_llm(prompt, model=GROUNDING_MODEL, max_tokens=120, temperature=0.0)
            self.llm_calls_used += 1
        except Exception as e:
            logger.error(f"LLM grounding failed for {goal}: {e}")
            return "unknown", 0.0

        # Parse "ANSWER: yes/no/unknown | CONFIDENCE: 0.X | EVIDENCE: ..." format
        answer, conf = "unknown", 0.3
        try:
            text_lower = text.lower()
            m_ans = re.search(r'answer:\s*(yes|no|unknown)', text_lower)
            if m_ans:
                answer = m_ans.group(1)
            else:
                if "yes" in text_lower[:100]:
                    answer = "yes"
                elif " no" in text_lower[:100] or text_lower[:10].startswith("no"):
                    answer = "no"
            m_conf = re.search(r'confidence:\s*([\d.]+)', text_lower)
            if m_conf:
                conf = max(0.0, min(1.0, float(m_conf.group(1))))
            else:
                conf = 0.7 if answer != "unknown" else 0.3
        except (ValueError, AttributeError):
            pass

        self._llm_cache[key] = (answer, conf)
        return answer, conf

    def solve(
        self,
        goals: list[Atom],
        sub: Substitution,
        depth: int,
        path_conf: float,
        node: Optional[ProofNode] = None,
    ) -> Optional[tuple[Substitution, float, ProofNode]]:
        """
        SLD resolution: try to prove goals under substitution sub.
        Returns (final_sub, confidence, proof_node) or None on failure.
        """
        if depth > self.max_depth:
            return None
        if not goals:
            if node:
                node.source = "success"
            return sub, path_conf, node or ProofNode("success", "success", path_conf)

        goal = apply_sub(goals[0], sub)
        rest = goals[1:]

        # Check if all args are ground
        all_ground = all(not a.is_var for a in goal.args)

        curr_node = ProofNode(str(goal), "searching", path_conf)

        # Try KB resolution
        for clause in self.kb:
            fresh = freshen(clause, self._counter)
            new_sub = unify(goal, fresh.head, sub)
            if new_sub is None:
                continue
            new_goals = fresh.body + rest
            child_node = ProofNode(str(goal), "kb", path_conf)
            result = self.solve(new_goals, new_sub, depth + 1, path_conf, child_node)
            if result is not None:
                final_sub, final_conf, proof_node = result
                curr_node.source = "rule" if fresh.body else "kb"
                curr_node.children.append(proof_node)
                curr_node.confidence = final_conf
                return final_sub, final_conf, curr_node

        # KB resolution failed → trigger LLM grounding (only for ground goals)
        if all_ground:
            answer, llm_conf = self._ground_with_llm(goal)
            llm_node = ProofNode(
                str(goal), "llm", llm_conf, llm_response=f"answer={answer}, conf={llm_conf:.2f}"
            )

            if answer == "yes":
                # Assert the fact and continue
                new_clause = Clause(head=deepcopy(goal))
                self.kb.append(new_clause)
                new_conf = path_conf * llm_conf
                child_node = ProofNode(str(goal), "llm_asserted", new_conf)
                result = self.solve(rest, sub, depth + 1, new_conf, child_node)
                if result is not None:
                    final_sub, final_conf, proof_node = result
                    curr_node.source = "llm"
                    curr_node.children.append(llm_node)
                    curr_node.children.append(proof_node)
                    curr_node.confidence = final_conf
                    return final_sub, final_conf, curr_node

            elif answer == "no":
                # Assert negation (closed-world)
                curr_node.source = "failed"
                curr_node.children.append(llm_node)
                return None

            else:  # unknown
                curr_node.source = "unknown"
                curr_node.children.append(llm_node)
                return None

        return None

    def query(self, goal: Atom) -> dict[str, Any]:
        """Run RFDE query and return result with proof trace."""
        t0 = time.time()
        root = ProofNode(str(goal), "root", 1.0)
        result = self.solve([goal], {}, depth=0, path_conf=1.0, node=root)

        if result is not None:
            _, conf, trace = result
            answer = "yes"
        else:
            conf = 0.0
            trace = root
            answer = "no"

        return {
            "answer": answer,
            "confidence": round(conf, 4),
            "llm_calls": self.llm_calls_used,
            "elapsed": round(time.time() - t0, 3),
            "proof_trace": _trace_to_dict(trace),
        }


def _trace_to_dict(node: ProofNode) -> dict[str, Any]:
    d: dict[str, Any] = {
        "goal": node.goal,
        "source": node.source,
        "confidence": round(node.confidence, 4),
    }
    if node.llm_response:
        d["llm_response"] = node.llm_response
    if node.children:
        d["children"] = [_trace_to_dict(c) for c in node.children]
    return d


def _trace_to_markdown(node: ProofNode, indent: int = 0) -> str:
    prefix = "  " * indent
    line = f"{prefix}- [{node.source}] {node.goal} (conf={node.confidence:.3f})"
    if node.llm_response:
        line += f" → {node.llm_response}"
    lines = [line]
    for child in node.children:
        lines.append(_trace_to_markdown(child, indent + 1))
    return "\n".join(lines)


# ─── Task definitions ─────────────────────────────────────────────────────────

@dataclass
class ReasoningTask:
    """A single reasoning task with document, query, and ground truth."""
    task_id: str
    dataset: str
    document: str
    query_atom: Atom
    query_nl: str  # natural-language form of query
    ground_truth: str  # "yes" or "no"
    background_rules: list[Clause] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_synthetic_tasks() -> list[ReasoningTask]:
    """Hand-coded synthetic reasoning tasks covering key RFDE scenarios."""
    tasks = []

    # Task S1: Direct fact (1-hop)
    tasks.append(ReasoningTask(
        task_id="S1",
        dataset="synthetic",
        document="Alice is the mother of Bob. Bob works as a teacher.",
        query_atom=_atom("mother", "alice", "bob"),
        query_nl="Is Alice the mother of Bob?",
        ground_truth="yes",
    ))

    # Task S2: Negation (fact not in document)
    tasks.append(ReasoningTask(
        task_id="S2",
        dataset="synthetic",
        document="Alice is the mother of Bob. Bob works as a teacher.",
        query_atom=_atom("mother", "alice", "charlie"),
        query_nl="Is Alice the mother of Charlie?",
        ground_truth="no",
    ))

    # Task S3: 2-hop family relation via rules
    # Rule: grandmother(X, Z) :- mother(X, Y), mother(Y, Z)
    # Rule: grandmother(X, Z) :- mother(X, Y), father(Y, Z)
    gm_rule1 = Clause(
        head=Atom("grandmother", [_var("X"), _var("Z")]),
        body=[
            Atom("mother", [_var("X"), _var("Y")]),
            Atom("mother", [_var("Y"), _var("Z")]),
        ],
    )
    gm_rule2 = Clause(
        head=Atom("grandmother", [_var("X"), _var("Z")]),
        body=[
            Atom("mother", [_var("X"), _var("Y")]),
            Atom("father", [_var("Y"), _var("Z")]),
        ],
    )
    tasks.append(ReasoningTask(
        task_id="S3",
        dataset="synthetic",
        document=(
            "Mary is the mother of John. John is the father of Emma. "
            "John is a carpenter who lives in Springfield."
        ),
        query_atom=_atom("grandmother", "mary", "emma"),
        query_nl="Is Mary the grandmother of Emma?",
        ground_truth="yes",
        background_rules=[gm_rule1, gm_rule2],
    ))

    # Task S4: 3-hop chain via transitivity rule
    # ancestor(X, Z) :- parent(X, Z)
    # ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
    anc_rule1 = Clause(
        head=Atom("ancestor", [_var("X"), _var("Z")]),
        body=[Atom("parent", [_var("X"), _var("Z")])],
    )
    anc_rule2 = Clause(
        head=Atom("ancestor", [_var("X"), _var("Z")]),
        body=[
            Atom("parent", [_var("X"), _var("Y")]),
            Atom("ancestor", [_var("Y"), _var("Z")]),
        ],
    )
    par_rule1 = Clause(
        head=Atom("parent", [_var("X"), _var("Y")]),
        body=[Atom("mother", [_var("X"), _var("Y")])],
    )
    par_rule2 = Clause(
        head=Atom("parent", [_var("X"), _var("Y")]),
        body=[Atom("father", [_var("X"), _var("Y")])],
    )
    tasks.append(ReasoningTask(
        task_id="S4",
        dataset="synthetic",
        document=(
            "Carol is the mother of David. David is the father of Eve. "
            "Eve is the mother of Frank. They all live in Boston."
        ),
        query_atom=_atom("ancestor", "carol", "frank"),
        query_nl="Is Carol an ancestor of Frank?",
        ground_truth="yes",
        background_rules=[anc_rule1, anc_rule2, par_rule1, par_rule2],
    ))

    # Task S5: Deductive rule (property inheritance)
    # expensive(X) :- color(X, red), valuable_color(red)
    exp_rule = Clause(
        head=Atom("expensive", [_var("X")]),
        body=[
            Atom("color", [_var("X"), _var("C")]),
            Atom("valuable_color", [_var("C")]),
        ],
    )
    tasks.append(ReasoningTask(
        task_id="S5",
        dataset="synthetic",
        document=(
            "The ball is red. Red objects are considered premium and expensive "
            "in this market. The box is blue."
        ),
        query_atom=_atom("expensive", "ball"),
        query_nl="Is the ball expensive?",
        ground_truth="yes",
        background_rules=[exp_rule],
    ))

    # Task S6: Hallucination test — query unsupported by document
    tasks.append(ReasoningTask(
        task_id="S6",
        dataset="synthetic",
        document="The cat sat on the mat. The dog barked loudly.",
        query_atom=_atom("owns", "alice", "cat"),
        query_nl="Does Alice own the cat?",
        ground_truth="no",
        metadata={"hallucination_test": True},
    ))

    # Task S7: Legal-style document
    tasks.append(ReasoningTask(
        task_id="S7",
        dataset="synthetic",
        document=(
            "In the matter of Smith v. Jones, the court found that defendant Jones "
            "breached the contract dated January 15, 2023. The plaintiff Smith suffered "
            "damages of $50,000 as a direct result of the breach. The court awarded "
            "compensatory damages to Smith."
        ),
        query_atom=_atom("liable", "jones", "breach"),
        query_nl="Is Jones liable for breach of contract?",
        ground_truth="yes",
    ))

    # Task S8: News article
    tasks.append(ReasoningTask(
        task_id="S8",
        dataset="synthetic",
        document=(
            "The city council voted 7-2 to approve the new housing development. "
            "Mayor Thompson signed the ordinance into law on Monday. "
            "The development will create 500 new housing units and 200 jobs. "
            "Environmental groups opposed the project."
        ),
        query_atom=_atom("approved", "housing_development", "council"),
        query_nl="Did the council approve the housing development?",
        ground_truth="yes",
    ))

    return tasks


def build_clutrr_tasks() -> list[ReasoningTask]:
    """
    CLUTRR-style family relation tasks (hand-crafted to match CLUTRR distribution).
    CLUTRR: Compositional Language Understanding with Text-based Relational Reasoning.
    """
    # Standard CLUTRR background rules for family relations
    rules = [
        # grandmother
        Clause(Atom("grandmother", [_var("X"), _var("Z")]),
               [Atom("mother", [_var("X"), _var("Y")]), Atom("mother", [_var("Y"), _var("Z")])]),
        Clause(Atom("grandmother", [_var("X"), _var("Z")]),
               [Atom("mother", [_var("X"), _var("Y")]), Atom("father", [_var("Y"), _var("Z")])]),
        # grandfather
        Clause(Atom("grandfather", [_var("X"), _var("Z")]),
               [Atom("father", [_var("X"), _var("Y")]), Atom("father", [_var("Y"), _var("Z")])]),
        Clause(Atom("grandfather", [_var("X"), _var("Z")]),
               [Atom("father", [_var("X"), _var("Y")]), Atom("mother", [_var("Y"), _var("Z")])]),
        # aunt
        Clause(Atom("aunt", [_var("X"), _var("Z")]),
               [Atom("sister", [_var("X"), _var("Y")]), Atom("parent", [_var("Y"), _var("Z")])]),
        # uncle
        Clause(Atom("uncle", [_var("X"), _var("Z")]),
               [Atom("brother", [_var("X"), _var("Y")]), Atom("parent", [_var("Y"), _var("Z")])]),
        # parent
        Clause(Atom("parent", [_var("X"), _var("Y")]),
               [Atom("mother", [_var("X"), _var("Y")])]),
        Clause(Atom("parent", [_var("X"), _var("Y")]),
               [Atom("father", [_var("X"), _var("Y")])]),
        # sibling
        Clause(Atom("sibling", [_var("X"), _var("Y")]),
               [Atom("brother", [_var("X"), _var("Y")])]),
        Clause(Atom("sibling", [_var("X"), _var("Y")]),
               [Atom("sister", [_var("X"), _var("Y")])]),
    ]

    tasks = []

    tasks.append(ReasoningTask(
        task_id="C1",
        dataset="clutrr",
        document=(
            "Sarah and her brother Mike grew up together. Mike later married Lisa, "
            "and they had a daughter named Sophie. Sarah works as a nurse."
        ),
        query_atom=_atom("aunt", "sarah", "sophie"),
        query_nl="Is Sarah the aunt of Sophie?",
        ground_truth="yes",
        background_rules=rules,
        metadata={"hops": 2},
    ))

    tasks.append(ReasoningTask(
        task_id="C2",
        dataset="clutrr",
        document=(
            "Tom is the father of Anna. Anna is the mother of Jake. "
            "Jake plays basketball every Saturday. Tom is retired."
        ),
        query_atom=_atom("grandfather", "tom", "jake"),
        query_nl="Is Tom the grandfather of Jake?",
        ground_truth="yes",
        background_rules=rules,
        metadata={"hops": 2},
    ))

    tasks.append(ReasoningTask(
        task_id="C3",
        dataset="clutrr",
        document=(
            "Helen is the mother of Peter. Peter married Kate, and Kate is the mother "
            "of Lucy. Helen loves to garden on weekends."
        ),
        query_atom=_atom("grandmother", "helen", "lucy"),
        query_nl="Is Helen the grandmother of Lucy?",
        ground_truth="yes",
        background_rules=rules,
        metadata={"hops": 2},
    ))

    tasks.append(ReasoningTask(
        task_id="C4",
        dataset="clutrr",
        document=(
            "James and his sister Rebecca were raised by their parents in Texas. "
            "James married Carol, who gave birth to their son Timothy. "
            "Rebecca is a teacher."
        ),
        query_atom=_atom("aunt", "rebecca", "timothy"),
        query_nl="Is Rebecca the aunt of Timothy?",
        ground_truth="yes",
        background_rules=rules,
        metadata={"hops": 2},
    ))

    tasks.append(ReasoningTask(
        task_id="C5",
        dataset="clutrr",
        document=(
            "George is the father of Henry. Henry and William are brothers. "
            "William fathered a child named Oliver. George likes fishing."
        ),
        query_atom=_atom("grandfather", "george", "oliver"),
        query_nl="Is George the grandfather of Oliver?",
        ground_truth="yes",
        background_rules=rules,
        metadata={"hops": 3},
    ))

    tasks.append(ReasoningTask(
        task_id="C6",
        dataset="clutrr",
        document=(
            "Diana is the daughter of Margaret. Margaret's sister is Elizabeth. "
            "Elizabeth has a son named Charles. Diana is an architect."
        ),
        query_atom=_atom("aunt", "elizabeth", "diana"),
        query_nl="Is Elizabeth the aunt of Diana?",
        ground_truth="yes",
        background_rules=rules,
        metadata={"hops": 2},
    ))

    tasks.append(ReasoningTask(
        task_id="C7",
        dataset="clutrr",
        document=(
            "Robert and Susan are siblings. Robert became a father when his daughter "
            "Patricia was born. Patricia now has a daughter named Linda."
        ),
        query_atom=_atom("aunt", "susan", "patricia"),
        query_nl="Is Susan the aunt of Patricia?",
        ground_truth="yes",
        background_rules=rules,
        metadata={"hops": 2},
    ))

    tasks.append(ReasoningTask(
        task_id="C8",
        dataset="clutrr",
        document=(
            "Alice's father is called David. David's father is called Frank. "
            "Alice studies computer science."
        ),
        query_atom=_atom("grandfather", "frank", "alice"),
        query_nl="Is Frank the grandfather of Alice?",
        ground_truth="yes",
        background_rules=rules,
        metadata={"hops": 2},
    ))

    return tasks


def build_ruletaker_tasks() -> list[ReasoningTask]:
    """
    RuleTaker-style deductive reasoning tasks.
    RuleTaker: Can a model reason about rules stated in natural language?
    Depth D1/D2: requires 1-2 rule applications.
    """
    tasks = []

    # D1 tasks (1-hop)
    tasks.append(ReasoningTask(
        task_id="R1",
        dataset="ruletaker",
        document=(
            "All mammals are warm-blooded. Dogs are mammals. "
            "Cats are mammals. Fish are not mammals."
        ),
        query_atom=_atom("warm_blooded", "dog"),
        query_nl="Is the dog warm-blooded?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("warm_blooded", [_var("X")]),
                   [Atom("mammal", [_var("X")])]),
        ],
        metadata={"depth": "D1"},
    ))

    tasks.append(ReasoningTask(
        task_id="R2",
        dataset="ruletaker",
        document=(
            "All students who study hard pass their exams. "
            "Maria studies hard every night. John does not study hard."
        ),
        query_atom=_atom("passes_exam", "maria"),
        query_nl="Does Maria pass her exam?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("passes_exam", [_var("X")]),
                   [Atom("studies_hard", [_var("X")])]),
        ],
        metadata={"depth": "D1"},
    ))

    tasks.append(ReasoningTask(
        task_id="R3",
        dataset="ruletaker",
        document=(
            "All things that are heavy sink in water. "
            "The rock is heavy. The feather is not heavy."
        ),
        query_atom=_atom("sinks", "rock"),
        query_nl="Does the rock sink in water?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("sinks", [_var("X")]),
                   [Atom("heavy", [_var("X")])]),
        ],
        metadata={"depth": "D1"},
    ))

    tasks.append(ReasoningTask(
        task_id="R4",
        dataset="ruletaker",
        document=(
            "All birds can fly. Penguins are birds. "
            "Eagles are birds. Eagles have sharp talons."
        ),
        query_atom=_atom("can_fly", "eagle"),
        query_nl="Can the eagle fly?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("can_fly", [_var("X")]),
                   [Atom("bird", [_var("X")])]),
        ],
        metadata={"depth": "D1"},
    ))

    # D2 tasks (2-hop)
    tasks.append(ReasoningTask(
        task_id="R5",
        dataset="ruletaker",
        document=(
            "All carnivores eat meat. All animals that eat meat are predators. "
            "Lions are carnivores. Rabbits are herbivores."
        ),
        query_atom=_atom("predator", "lion"),
        query_nl="Is the lion a predator?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("eats_meat", [_var("X")]),
                   [Atom("carnivore", [_var("X")])]),
            Clause(Atom("predator", [_var("X")]),
                   [Atom("eats_meat", [_var("X")])]),
        ],
        metadata={"depth": "D2"},
    ))

    tasks.append(ReasoningTask(
        task_id="R6",
        dataset="ruletaker",
        document=(
            "Athletes who train daily are in good shape. "
            "People in good shape are healthy. "
            "Alex trains every single day without exception."
        ),
        query_atom=_atom("healthy", "alex"),
        query_nl="Is Alex healthy?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("in_good_shape", [_var("X")]),
                   [Atom("trains_daily", [_var("X")])]),
            Clause(Atom("healthy", [_var("X")]),
                   [Atom("in_good_shape", [_var("X")])]),
        ],
        metadata={"depth": "D2"},
    ))

    tasks.append(ReasoningTask(
        task_id="R7",
        dataset="ruletaker",
        document=(
            "All renewable energy sources are sustainable. "
            "Sustainable resources protect the environment. "
            "Solar power is a renewable energy source."
        ),
        query_atom=_atom("protects_environment", "solar_power"),
        query_nl="Does solar power protect the environment?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("sustainable", [_var("X")]),
                   [Atom("renewable_energy", [_var("X")])]),
            Clause(Atom("protects_environment", [_var("X")]),
                   [Atom("sustainable", [_var("X")])]),
        ],
        metadata={"depth": "D2"},
    ))

    tasks.append(ReasoningTask(
        task_id="R8",
        dataset="ruletaker",
        document=(
            "Anyone who commits fraud is dishonest. "
            "Dishonest people cannot be trusted. "
            "Victor committed fraud last year according to court records."
        ),
        query_atom=_atom("cannot_be_trusted", "victor"),
        query_nl="Cannot Victor be trusted?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("dishonest", [_var("X")]),
                   [Atom("commits_fraud", [_var("X")])]),
            Clause(Atom("cannot_be_trusted", [_var("X")]),
                   [Atom("dishonest", [_var("X")])]),
        ],
        metadata={"depth": "D2"},
    ))

    tasks.append(ReasoningTask(
        task_id="R9",
        dataset="ruletaker",
        document=(
            "All planets orbit a star. Objects that orbit a star are part of a solar system. "
            "Mars is a planet."
        ),
        query_atom=_atom("part_of_solar_system", "mars"),
        query_nl="Is Mars part of a solar system?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("orbits_star", [_var("X")]),
                   [Atom("planet", [_var("X")])]),
            Clause(Atom("part_of_solar_system", [_var("X")]),
                   [Atom("orbits_star", [_var("X")])]),
        ],
        metadata={"depth": "D2"},
    ))

    tasks.append(ReasoningTask(
        task_id="R10",
        dataset="ruletaker",
        document=(
            "All electric vehicles have zero direct emissions. "
            "The Tesla Model S is an electric vehicle. "
            "Gasoline cars have emissions."
        ),
        query_atom=_atom("zero_direct_emissions", "tesla_model_s"),
        query_nl="Does the Tesla Model S have zero direct emissions?",
        ground_truth="yes",
        background_rules=[
            Clause(Atom("zero_direct_emissions", [_var("X")]),
                   [Atom("electric_vehicle", [_var("X")])]),
        ],
        metadata={"depth": "D1"},
    ))

    return tasks


# ─── RFDE Runner ─────────────────────────────────────────────────────────────

def run_rfde(task: ReasoningTask) -> dict[str, Any]:
    """Run RFDE on a single task."""
    engine = RFDEEngine(document=task.document, max_depth=12, max_llm_calls=15)

    # Load background rules
    for rule in task.background_rules:
        engine.assert_rule(rule.head, rule.body)

    t0 = time.time()
    try:
        result = engine.query(task.query_atom)
    except Exception as e:
        logger.error(f"RFDE error on {task.task_id}: {e}")
        result = {"answer": "error", "confidence": 0.0, "llm_calls": 0, "elapsed": 0, "proof_trace": {}}

    result["elapsed"] = round(time.time() - t0, 3)
    result["task_id"] = task.task_id
    result["correct"] = result["answer"] == task.ground_truth
    result["proof_trace_md"] = _trace_to_markdown(
        ProofNode(
            str(task.query_atom),
            result.get("proof_trace", {}).get("source", "unknown"),
            result.get("confidence", 0.0)
        )
    )
    return result


# ─── Baselines ────────────────────────────────────────────────────────────────

def run_cot_baseline(task: ReasoningTask) -> dict[str, Any]:
    """Baseline 1: Chain-of-Thought LLM reasoning over the full document."""
    rules_desc = ""
    if task.background_rules:
        rule_strs = [str(r) for r in task.background_rules]
        rules_desc = f"\n\nBackground rules:\n" + "\n".join(rule_strs)

    prompt = (
        f"Document:\n\"{task.document}\"{rules_desc}\n\n"
        f"Question: {task.query_nl}\n\n"
        f"Think step by step. Then answer with EXACTLY:\n"
        f'{{\"answer\": \"yes\" or \"no\", \"reasoning\": \"brief\"}}'
    )

    t0 = time.time()
    try:
        text = call_llm(prompt, model=BASELINE_MODEL, max_tokens=300, temperature=0.0)
    except Exception as e:
        logger.error(f"CoT baseline error on {task.task_id}: {e}")
        return {"answer": "error", "correct": False, "elapsed": time.time() - t0}

    # Parse
    answer = "unknown"
    try:
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            answer = parsed.get("answer", "unknown").lower().strip()
        else:
            text_lower = text.lower()
            if "yes" in text_lower[-200:]:
                answer = "yes"
            elif "no" in text_lower[-200:]:
                answer = "no"
    except (json.JSONDecodeError, ValueError):
        text_lower = text.lower()
        if "yes" in text_lower[-100:]:
            answer = "yes"
        elif "no" in text_lower[-100:]:
            answer = "no"

    return {
        "answer": answer,
        "correct": answer == task.ground_truth,
        "elapsed": round(time.time() - t0, 3),
        "raw_response": text[:500],
    }


def run_rag_baseline(task: ReasoningTask) -> dict[str, Any]:
    """Baseline 2: BM25 retrieval + LLM answer."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    # Tokenize document into sentences
    try:
        sentences = nltk.sent_tokenize(task.document)
    except Exception:
        sentences = task.document.split(". ")

    if not sentences:
        sentences = [task.document]

    # BM25 retrieval
    try:
        tokenized = [s.lower().split() for s in sentences]
        bm25 = BM25Okapi(tokenized)
        query_tokens = task.query_nl.lower().split()
        scores = bm25.get_scores(query_tokens)
        top_k = min(3, len(sentences))
        top_idx = np.argsort(scores)[::-1][:top_k]
        retrieved = " ".join(sentences[i] for i in sorted(top_idx))
    except Exception:
        retrieved = task.document  # fallback: use full document

    rules_desc = ""
    if task.background_rules:
        rule_strs = [str(r) for r in task.background_rules]
        rules_desc = f"\n\nBackground rules:\n" + "\n".join(rule_strs)

    prompt = (
        f"Retrieved facts:\n\"{retrieved}\"{rules_desc}\n\n"
        f"Question: {task.query_nl}\n\n"
        f"Based ONLY on the retrieved facts. Answer:\n"
        f'{{\"answer\": \"yes\" or \"no\"}}'
    )

    t0 = time.time()
    try:
        text = call_llm(prompt, model=BASELINE_MODEL, max_tokens=80, temperature=0.0)
    except Exception as e:
        logger.error(f"RAG baseline error on {task.task_id}: {e}")
        return {"answer": "error", "correct": False, "elapsed": time.time() - t0}

    answer = "unknown"
    try:
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            answer = parsed.get("answer", "unknown").lower().strip()
        else:
            text_lower = text.lower()
            answer = "yes" if "yes" in text_lower[:100] else "no"
    except (json.JSONDecodeError, ValueError):
        text_lower = text.lower()
        answer = "yes" if "yes" in text_lower[:100] else "no"

    return {
        "answer": answer,
        "correct": answer == task.ground_truth,
        "elapsed": round(time.time() - t0, 3),
        "retrieved_context": retrieved[:300],
    }


def run_linc_baseline(task: ReasoningTask) -> dict[str, Any]:
    """
    Baseline 3: LINC-style eager FOL translation.
    LLM translates entire document to Prolog facts upfront, then we run resolution.
    """
    rules_desc = ""
    if task.background_rules:
        rule_strs = [str(r) for r in task.background_rules]
        rules_desc = f"\n\nBackground rules (already provided):\n" + "\n".join(rule_strs)

    pred_hint = task.query_atom.pred
    args_hint = ", ".join(str(a) for a in task.query_atom.args)

    prompt = (
        f"Document:\n\"{task.document}\"{rules_desc}\n\n"
        f"Extract ALL atomic facts from this document as Prolog facts.\n"
        f"Use predicate names like: mother(X,Y), father(X,Y), sister(X,Y), brother(X,Y), "
        f"mammal(X), bird(X), carnivore(X), warm_blooded(X), and similar.\n"
        f"The target query is: {pred_hint}({args_hint})\n"
        f"Use lowercase with underscores for all names.\n\n"
        f"Respond with ONLY valid JSON:\n"
        f'{{\"facts\": [\"fact1(arg1,arg2)\", \"fact2(arg1)\", ...]}}'
    )

    t0 = time.time()
    try:
        text = call_llm(prompt, model=BASELINE_MODEL, max_tokens=400, temperature=0.0)
    except Exception as e:
        logger.error(f"LINC baseline LLM error on {task.task_id}: {e}")
        return {"answer": "error", "correct": False, "elapsed": time.time() - t0}

    # Parse extracted facts
    extracted_facts: list[str] = []
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            extracted_facts = parsed.get("facts", [])
    except (json.JSONDecodeError, ValueError):
        # Try regex extraction of fact(arg) patterns
        extracted_facts = re.findall(r'\w+\([^)]+\)', text)

    # Parse facts into Clauses and run resolution
    linc_engine = RFDEEngine(document="", max_depth=12, max_llm_calls=0)  # NO LLM calls

    # Load background rules
    for rule in task.background_rules:
        linc_engine.assert_rule(rule.head, rule.body)

    # Assert extracted facts
    fact_parse_errors = 0
    for fact_str in extracted_facts:
        m = re.match(r'(\w+)\(([^)]+)\)', fact_str.strip())
        if m:
            pred = m.group(1)
            args = [a.strip() for a in m.group(2).split(",")]
            try:
                linc_engine.assert_fact(pred, *args)
            except Exception:
                fact_parse_errors += 1
        else:
            fact_parse_errors += 1

    # Try resolution without LLM
    try:
        result = linc_engine.query(task.query_atom)
        answer = result["answer"]
    except Exception:
        answer = "error"

    return {
        "answer": answer,
        "correct": answer == task.ground_truth,
        "elapsed": round(time.time() - t0, 3),
        "extracted_facts": extracted_facts[:20],
        "fact_parse_errors": fact_parse_errors,
        "extracted_fact_count": len(extracted_facts),
    }


# ─── Hallucination counting ───────────────────────────────────────────────────

def count_hallucinations_cot(response: str, document: str) -> int:
    """
    Rough hallucination count: facts in CoT reasoning not supported by the document.
    Uses a simple entity/relation overlap heuristic.
    """
    doc_words = set(document.lower().split())
    response_words = response.lower().split()
    # Find named entities in response not in doc
    hallucinated = 0
    for w in response_words:
        w_clean = re.sub(r'[^a-z]', '', w)
        if len(w_clean) > 3 and w_clean[0].isupper() if w else False:
            if w_clean.lower() not in doc_words:
                hallucinated += 1
    return hallucinated


# ─── Metrics aggregation ─────────────────────────────────────────────────────

def compute_metrics(results: list[dict], method: str) -> dict[str, float]:
    """Compute accuracy and related metrics for a method's results."""
    if not results:
        return {}
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    accuracy = correct / total

    # Count by dataset
    by_dataset: dict[str, list[bool]] = {}
    for r in results:
        ds = r.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(r.get("correct", False))

    metrics: dict[str, float] = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
    }
    for ds, bools in by_dataset.items():
        metrics[f"accuracy_{ds}"] = round(sum(bools) / len(bools), 4)

    # Avg LLM calls (RFDE only)
    llm_calls = [r.get("llm_calls", 0) for r in results]
    if any(c > 0 for c in llm_calls):
        metrics["avg_llm_calls"] = round(sum(llm_calls) / len(llm_calls), 2)

    # Avg latency
    latencies = [r.get("elapsed", 0) for r in results]
    metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)

    logger.info(f"[{method}] accuracy={accuracy:.1%} ({correct}/{total})")
    return metrics


# ─── Main experiment loop ─────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main() -> None:
    ws = Path("/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run_vlVwS0MntEIr/3_invention_loop/iter_1/gen_art/gen_art_experiment_1")
    ws.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RFDE Experiment: Resolution-Failure-Directed Extraction")
    logger.info("=" * 60)

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    # Verify API key works with a test call
    logger.info("Testing OpenRouter API connection...")
    test_resp = call_llm("Say 'ok'", model=GROUNDING_MODEL, max_tokens=5)
    logger.info(f"API test response: {test_resp!r}")

    # Build all tasks
    logger.info("Building task sets...")
    synthetic_tasks = build_synthetic_tasks()
    clutrr_tasks = build_clutrr_tasks()
    ruletaker_tasks = build_ruletaker_tasks()
    all_tasks = synthetic_tasks + clutrr_tasks + ruletaker_tasks
    logger.info(f"Tasks: {len(synthetic_tasks)} synthetic, {len(clutrr_tasks)} CLUTRR, {len(ruletaker_tasks)} RuleTaker")

    # ─── Run experiments ─────────────────────────────────────────────────────
    rfde_results: list[dict] = []
    cot_results: list[dict] = []
    rag_results: list[dict] = []
    linc_results: list[dict] = []

    proof_traces: list[dict] = []  # rich traces for first 5 tasks

    for i, task in enumerate(all_tasks):
        if COST_TRACKER["total"] >= HARD_BUDGET:
            logger.warning(f"Budget exhausted at task {i}. Stopping.")
            break

        logger.info(f"\n--- Task {task.task_id} ({task.dataset}) | budget ${COST_TRACKER['total']:.4f}/${HARD_BUDGET}")
        logger.info(f"    Query: {task.query_nl}")
        logger.info(f"    GT: {task.ground_truth}")

        # RFDE
        rfde_res = run_rfde(task)
        rfde_res["dataset"] = task.dataset
        rfde_results.append(rfde_res)
        logger.info(f"    RFDE: {rfde_res['answer']} (conf={rfde_res.get('confidence',0):.2f}, "
                    f"llm_calls={rfde_res.get('llm_calls',0)}, correct={rfde_res['correct']})")

        # Save rich proof trace for first 5 tasks
        if i < 5:
            proof_traces.append({
                "task_id": task.task_id,
                "query": task.query_nl,
                "document": task.document[:300],
                "ground_truth": task.ground_truth,
                "rfde_answer": rfde_res["answer"],
                "rfde_confidence": rfde_res.get("confidence", 0),
                "proof_trace": rfde_res.get("proof_trace", {}),
                "proof_trace_markdown": rfde_res.get("proof_trace_md", ""),
            })

        if COST_TRACKER["total"] >= HARD_BUDGET:
            break

        # CoT baseline
        cot_res = run_cot_baseline(task)
        cot_res["dataset"] = task.dataset
        cot_res["task_id"] = task.task_id
        cot_results.append(cot_res)
        logger.info(f"    CoT: {cot_res['answer']} (correct={cot_res['correct']})")

        if COST_TRACKER["total"] >= HARD_BUDGET:
            break

        # RAG baseline
        rag_res = run_rag_baseline(task)
        rag_res["dataset"] = task.dataset
        rag_res["task_id"] = task.task_id
        rag_results.append(rag_res)
        logger.info(f"    RAG: {rag_res['answer']} (correct={rag_res['correct']})")

        if COST_TRACKER["total"] >= HARD_BUDGET:
            break

        # LINC baseline
        linc_res = run_linc_baseline(task)
        linc_res["dataset"] = task.dataset
        linc_res["task_id"] = task.task_id
        linc_results.append(linc_res)
        logger.info(f"    LINC: {linc_res['answer']} (correct={linc_res['correct']}, "
                    f"facts={linc_res.get('extracted_fact_count',0)})")

        gc.collect()

    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment complete. Total cost: ${COST_TRACKER['total']:.4f} ({COST_TRACKER['calls']} LLM calls)")

    # ─── Metrics ─────────────────────────────────────────────────────────────
    n = min(len(rfde_results), len(cot_results), len(rag_results), len(linc_results))
    logger.info(f"\nComputing metrics on {n} completed tasks...")

    rfde_metrics = compute_metrics(rfde_results[:n], "RFDE")
    cot_metrics = compute_metrics(cot_results[:n], "CoT")
    rag_metrics = compute_metrics(rag_results[:n], "RAG")
    linc_metrics = compute_metrics(linc_results[:n], "LINC")

    # Hallucination estimation:
    # RFDE only calls LLM for specific predicates → bounded hallucination
    # Estimate: each LLM call that returns "yes" for an unsupported predicate = 1 hallucination
    # For baselines: count unsupported statements in free-form responses

    # RFDE hallucination rate: fraction of LLM-asserted facts that were "wrong" answers
    rfde_halluc = sum(
        1 for r in rfde_results[:n]
        if r.get("llm_calls", 0) > 0 and not r.get("correct", False)
    ) / max(1, len([r for r in rfde_results[:n] if r.get("llm_calls", 0) > 0]))

    # CoT: wrong + over-confident = potential hallucination proxy
    cot_halluc = sum(1 for r in cot_results[:n] if not r.get("correct", False)) / max(1, n)
    rag_halluc = sum(1 for r in rag_results[:n] if not r.get("correct", False)) / max(1, n)
    linc_halluc = sum(1 for r in linc_results[:n] if not r.get("correct", False)) / max(1, n)

    halluc_reduction_vs_cot = max(0, cot_halluc - rfde_halluc)
    halluc_reduction_vs_cot_pct = (
        halluc_reduction_vs_cot / max(0.01, cot_halluc) * 100
    )

    logger.info(f"\nHallucination rate (error proxy):")
    logger.info(f"  RFDE:  {rfde_halluc:.1%}")
    logger.info(f"  CoT:   {cot_halluc:.1%}")
    logger.info(f"  RAG:   {rag_halluc:.1%}")
    logger.info(f"  LINC:  {linc_halluc:.1%}")
    logger.info(f"  RFDE vs CoT reduction: {halluc_reduction_vs_cot_pct:.1f}%")

    # LINC atomic fact extraction precision/recall
    linc_fact_counts = [r.get("extracted_fact_count", 0) for r in linc_results[:n]]
    avg_linc_facts = sum(linc_fact_counts) / max(1, len(linc_fact_counts))
    linc_correct = sum(1 for r in linc_results[:n] if r.get("correct", False))

    # RFDE average LLM calls
    rfde_calls_list = [r.get("llm_calls", 0) for r in rfde_results[:n]]
    avg_rfde_calls = sum(rfde_calls_list) / max(1, len(rfde_calls_list))

    # ─── Build method_out.json ────────────────────────────────────────────────
    all_examples = []
    for i in range(min(len(rfde_results), len(cot_results), len(rag_results), len(linc_results))):
        task = all_tasks[i]
        rfde_r = rfde_results[i]
        cot_r = cot_results[i]
        rag_r = rag_results[i]
        linc_r = linc_results[i]

        proof_trace_str = json.dumps(rfde_r.get("proof_trace", {}), indent=2)

        example: dict[str, Any] = {
            "input": (
                f"[{task.dataset.upper()} | {task.task_id}] Document: {task.document} "
                f"| Query: {task.query_nl}"
            ),
            "output": task.ground_truth,
            "predict_rfde": rfde_r["answer"],
            "predict_cot": cot_r["answer"],
            "predict_rag": rag_r["answer"],
            "predict_linc": linc_r["answer"],
            "metadata_task_id": task.task_id,
            "metadata_dataset": task.dataset,
            "metadata_rfde_correct": str(rfde_r.get("correct", False)),
            "metadata_cot_correct": str(cot_r.get("correct", False)),
            "metadata_rag_correct": str(rag_r.get("correct", False)),
            "metadata_linc_correct": str(linc_r.get("correct", False)),
            "metadata_rfde_confidence": str(round(rfde_r.get("confidence", 0.0), 4)),
            "metadata_rfde_llm_calls": str(rfde_r.get("llm_calls", 0)),
            "metadata_rfde_latency_s": str(rfde_r.get("elapsed", 0)),
            "metadata_linc_fact_count": str(linc_r.get("extracted_fact_count", 0)),
            "metadata_proof_trace": proof_trace_str[:2000],
        }

        # Add depth/hop metadata if available
        if task.metadata.get("hops"):
            example["metadata_hops"] = str(task.metadata["hops"])
        if task.metadata.get("depth"):
            example["metadata_depth"] = task.metadata["depth"]

        all_examples.append(example)

    # Summary metrics as a dataset entry
    summary_example: dict[str, Any] = {
        "input": "EXPERIMENT SUMMARY: Aggregated metrics across all tasks",
        "output": "see metadata",
        "predict_rfde": f"accuracy={rfde_metrics.get('accuracy', 0):.4f}",
        "predict_cot": f"accuracy={cot_metrics.get('accuracy', 0):.4f}",
        "predict_rag": f"accuracy={rag_metrics.get('accuracy', 0):.4f}",
        "predict_linc": f"accuracy={linc_metrics.get('accuracy', 0):.4f}",
        "metadata_rfde_accuracy": str(rfde_metrics.get("accuracy", 0)),
        "metadata_cot_accuracy": str(cot_metrics.get("accuracy", 0)),
        "metadata_rag_accuracy": str(rag_metrics.get("accuracy", 0)),
        "metadata_linc_accuracy": str(linc_metrics.get("accuracy", 0)),
        "metadata_rfde_halluc_rate": str(round(rfde_halluc, 4)),
        "metadata_cot_halluc_rate": str(round(cot_halluc, 4)),
        "metadata_halluc_reduction_pct": str(round(halluc_reduction_vs_cot_pct, 2)),
        "metadata_avg_rfde_llm_calls": str(round(avg_rfde_calls, 2)),
        "metadata_avg_linc_facts_extracted": str(round(avg_linc_facts, 2)),
        "metadata_total_llm_cost_usd": str(round(COST_TRACKER["total"], 4)),
        "metadata_total_llm_calls": str(COST_TRACKER["calls"]),
        "metadata_tasks_completed": str(n),
    }

    method_out = {
        "metadata": {
            "method_name": "RFDE (Resolution-Failure-Directed Extraction)",
            "description": (
                "Neuro-symbolic pipeline: backward-chaining SLD resolution with "
                "LLM-triggered fact grounding on resolution failure. "
                "Baselines: CoT, BM25-RAG, LINC-style eager FOL translation."
            ),
            "parameters": {
                "grounding_model": GROUNDING_MODEL,
                "baseline_model": BASELINE_MODEL,
                "max_depth": 12,
                "max_llm_calls_per_query": 15,
                "confidence_aggregation": "product_rule",
                "resolution_strategy": "SLD_backward_chaining",
            },
            "results_summary": {
                "rfde_accuracy": rfde_metrics.get("accuracy", 0),
                "cot_accuracy": cot_metrics.get("accuracy", 0),
                "rag_accuracy": rag_metrics.get("accuracy", 0),
                "linc_accuracy": linc_metrics.get("accuracy", 0),
                "rfde_hallucination_rate": round(rfde_halluc, 4),
                "cot_hallucination_rate": round(cot_halluc, 4),
                "hallucination_reduction_vs_cot_pct": round(halluc_reduction_vs_cot_pct, 2),
                "avg_rfde_llm_calls_per_query": round(avg_rfde_calls, 2),
                "total_llm_cost_usd": round(COST_TRACKER["total"], 4),
                "total_llm_calls": COST_TRACKER["calls"],
                "tasks_completed": n,
            },
            "proof_traces": proof_traces[:5],
            "dataset_breakdown": {
                "rfde": {
                    ds: rfde_metrics.get(f"accuracy_{ds}", None)
                    for ds in ["synthetic", "clutrr", "ruletaker"]
                },
                "cot": {
                    ds: cot_metrics.get(f"accuracy_{ds}", None)
                    for ds in ["synthetic", "clutrr", "ruletaker"]
                },
                "rag": {
                    ds: rag_metrics.get(f"accuracy_{ds}", None)
                    for ds in ["synthetic", "clutrr", "ruletaker"]
                },
                "linc": {
                    ds: linc_metrics.get(f"accuracy_{ds}", None)
                    for ds in ["synthetic", "clutrr", "ruletaker"]
                },
            },
        },
        "datasets": [
            {
                "dataset": "rfde_experiment_all_tasks",
                "examples": all_examples,
            },
            {
                "dataset": "experiment_summary",
                "examples": [summary_example],
            },
        ],
    }

    out_path = ws / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2))
    logger.info(f"\nSaved method_out.json: {len(all_examples)} task results + summary")
    logger.info(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    # Print final comparison table
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS TABLE")
    logger.info("="*60)
    logger.info(f"{'Method':<12} {'Accuracy':>10} {'Halluc%':>10} {'Avg LLM calls':>15}")
    logger.info("-"*50)
    logger.info(f"{'RFDE':<12} {rfde_metrics.get('accuracy',0):>10.1%} {rfde_halluc:>10.1%} {avg_rfde_calls:>15.1f}")
    logger.info(f"{'CoT':<12} {cot_metrics.get('accuracy',0):>10.1%} {cot_halluc:>10.1%} {'1 (full doc)':>15}")
    logger.info(f"{'RAG+BM25':<12} {rag_metrics.get('accuracy',0):>10.1%} {rag_halluc:>10.1%} {'1 (top-3)':>15}")
    logger.info(f"{'LINC':<12} {linc_metrics.get('accuracy',0):>10.1%} {linc_halluc:>10.1%} {'1 (translate)':>15}")
    logger.info("="*60)
    logger.info(f"Total cost: ${COST_TRACKER['total']:.4f} ({COST_TRACKER['calls']} calls)")


if __name__ == "__main__":
    main()
