#!/usr/bin/env python3
"""Load and standardize neuro-symbolic reasoning datasets to exp_sel_data_out schema."""

import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/data_run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"


def load_json(path: Path) -> list:
    logger.info(f"Loading {path.name} ({path.stat().st_size // 1024}KB)")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"{path.name}: standard JSON parse failed, trying line-by-line fallback")
        rows = []
        skipped = 0
        with path.open() as f:
            for line in f:
                line = line.strip().rstrip(",")
                if not line or line in ("[", "]"):
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
        if skipped:
            logger.warning(f"{path.name}: skipped {skipped} unparseable lines")
        return rows


def process_clutrr(rows: list) -> list:
    """CLUTRR: kinship multi-hop reasoning stories."""
    examples = []
    for row in rows:
        query_raw = row.get("query", "")
        # parse "('Ashley', 'Nicholas')" -> "Ashley" and "Nicholas"
        names = query_raw.strip("()").replace("'", "").split(", ")
        name1 = names[0] if len(names) > 0 else "?"
        name2 = names[1] if len(names) > 1 else "?"

        story = row.get("story") or row.get("clean_story", "")
        inp = (
            f"Story: {story}\n"
            f"Query: What is the kinship relation between {name1} and {name2}?"
        )
        out = str(row.get("target_text", ""))

        # hop depth from edge count
        edge_types_raw = row.get("edge_types", "[]")
        try:
            edge_types = json.loads(edge_types_raw.replace("'", '"'))
            hop_depth = len(edge_types)
        except Exception:
            hop_depth = 0

        examples.append({
            "input": inp,
            "output": out,
            "metadata_source_id": str(row.get("id", "")),
            "metadata_hop_depth": hop_depth,
            "metadata_task_name": str(row.get("task_name", "")),
            "metadata_relation_chain": str(row.get("f_comb", "")),
            "metadata_proof_state": str(row.get("proof_state", ""))[:500],
            "metadata_task_type": "kinship_classification",
        })
    return examples


def process_ruletaker(rows: list) -> list:
    """RuleTaker: logical entailment from context + rules."""
    examples = []
    for row in rows:
        context = row.get("context", "")
        question = row.get("question", "")
        label = row.get("label", "")
        config = row.get("config", "")

        inp = f"Context: {context}\nStatement: {question}\nDoes the context entail the statement?"
        out = label  # "entailment" or "not entailment"

        # derive hop depth from config name like "depth-3"
        hop_depth = 0
        if config.startswith("depth-"):
            try:
                hop_depth = int(config.split("-")[1])
            except ValueError:
                pass

        examples.append({
            "input": inp,
            "output": out,
            "metadata_config": config,
            "metadata_hop_depth": hop_depth,
            "metadata_task_type": "logical_entailment",
        })
    return examples


def process_proofwriter(rows: list) -> list:
    """ProofWriter: proof generation + QA over theory."""
    examples = []
    for row in rows:
        theory = row.get("theory", "")
        question = row.get("question", "")
        answer = str(row.get("answer", ""))
        config = row.get("config", "")
        q_dep = row.get("QDep", 0)

        inp = f"Theory: {theory}\nQuestion: {question}\nAnswer True or False."
        out = answer  # "True" or "False"

        examples.append({
            "input": inp,
            "output": out,
            "metadata_source_id": str(row.get("id", "")),
            "metadata_hop_depth": int(q_dep) if q_dep is not None else 0,
            "metadata_config": config,
            "metadata_n_facts": int(row.get("NFact", 0)) if row.get("NFact") is not None else 0,
            "metadata_n_rules": int(row.get("NRule", 0)) if row.get("NRule") is not None else 0,
            "metadata_task_type": "proof_question_answering",
        })
    return examples


def process_ruletaker_d5(rows: list) -> list:
    """RuleTaker-d5-70k: 5-hop depth compact format."""
    examples = []
    for row in rows:
        question = row.get("question", "")
        answer_raw = row.get("answer", [""])
        # answer is a list like ["true"] or ["false"]
        if isinstance(answer_raw, list):
            answer = answer_raw[0] if answer_raw else ""
        else:
            answer = str(answer_raw)

        examples.append({
            "input": question,
            "output": answer,
            "metadata_hop_depth": 5,
            "metadata_task_type": "logical_entailment_5hop",
        })
    return examples


@logger.catch(reraise=True)
def main() -> None:
    Path("logs").mkdir(exist_ok=True)

    dataset_files = {
        "CLUTRR_v1": DATASETS_DIR / "full_CLUTRR_v1_gen_train234_test2to10_train.json",
        "RuleTaker": DATASETS_DIR / "full_tasksource_ruletaker_default_train.json",
        "ProofWriter": DATASETS_DIR / "full_tasksource_proofwriter_default_train.json",
    }

    processors = {
        "CLUTRR_v1": process_clutrr,
        "RuleTaker": process_ruletaker,
        "ProofWriter": process_proofwriter,
    }

    datasets_out = []
    for name, path in dataset_files.items():
        if not path.exists():
            logger.warning(f"Skipping {name}: file not found at {path}")
            continue
        try:
            rows = load_json(path)
            logger.info(f"{name}: {len(rows)} rows loaded")
            examples = processors[name](rows)
            logger.info(f"{name}: {len(examples)} examples produced")
            datasets_out.append({"dataset": name, "examples": examples})
        except Exception:
            logger.error(f"Failed processing {name}")
            raise

    output = {"datasets": datasets_out}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    total = sum(len(d["examples"]) for d in datasets_out)
    logger.info(f"Saved {total} total examples across {len(datasets_out)} datasets → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
