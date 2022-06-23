import sys
import argparse
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List

from anntools import Collection, DisjointSet, Sentence

CORRECT_A = "correct_A"
INCORRECT_A = "incorrect_A"
PARTIAL_A = "partial_A"
SPURIOUS_A = "spurious_A"
MISSING_A = "missing_A"

CORRECT_B = "correct_B"
SPURIOUS_B = "spurious_B"
MISSING_B = "missing_B"

SAME_AS = "same-as"


SCENARIOS = {
    1: ("scenario1-main", False, False),
    2: ("scenario2-taskA", False, True),
    3: ("scenario3-taskB", True, False),
}

def score_main(gold, submit, verbose, scenarios, runs, prefix):
    print("#####################")
    runs_data = {}

    for run in runs:
        run_data = {}

        if not (submit / f"run{run}").exists():
            print(f"Run {run} not found!")
            continue

        for id in scenarios:
            folder, skipA, skipB = SCENARIOS[id]

            print(f"Scoring scenario {id} on run {run}:\n")
            run_data[folder.split("-")[0]] = main_scenario(gold / folder / "output.txt", submit / f"run{run}" / folder / "output.txt", skipA, skipB, verbose)

        runs_data[f"run{run}"] = run_data


    report_main(runs_data, prefix)


def report_main(runs_data, prefix):
    keys = {f"scenario{s}_{metric}": 0 for s in [1, 2, 3] for metric in ["f1", "precision", "recall", "best"]}

    for run_id, run_data in runs_data.items():
        for scn_id, scn_data in run_data.items():
            if scn_data["f1"] <= keys[f"{scn_id}_f1"]:
                pass

            for metric in scn_data:
                keys[f"{scn_id}_{metric}"] = scn_data[metric]

            keys[f"{scn_id}_best"] = run_id

    for k, v in keys.items():
        print(f"{prefix}{k}:{v}")


def main_scenario(gold_input, submit_input, skip_A, skip_B, verbose):
    gold = Collection()
    gold.load(gold_input)

    submit = Collection()
    submit.load(submit_input)

    data = OrderedDict()

    dataA = subtaskA(gold, submit, verbose)
    data.update(dataA)
    if not skip_A:
        report(dataA, verbose)

    if not skip_B:
        dataB = subtaskB(gold, submit, dataA, verbose)
        data.update(dataB)
        report(dataB, verbose)

    print("-" * 20)

    metrics = compute_metrics(data, skip_A, skip_B)

    for key, value in metrics.items():
        print("{0}: {1:0.4}".format(key, value))

    return metrics
def compute_metrics(data, skipA=False, skipB=False):
    correct = 0
    partial = 0
    incorrect = 0
    missing = 0
    spurious = 0

    if not skipA:
        correct += len(data[CORRECT_A])
        incorrect += len(data[INCORRECT_A])
        partial += len(data[PARTIAL_A])
        missing += len(data[MISSING_A])
        spurious += len(data[SPURIOUS_A])

    if not skipB:
        correct += len(data[CORRECT_B])
        missing += len(data[MISSING_B])
        spurious += len(data[SPURIOUS_B])

    recall_num = correct + 0.5 * partial
    recall_den = correct + partial + incorrect + missing
    recall = recall_num / recall_den if recall_den > 0 else 0.

    precision_num = correct + 0.5 * partial
    precision_den = correct + partial + incorrect + spurious
    precision = precision_num / precision_den if precision_den > 0 else 0.

    f1_num = 2 * recall * precision
    f1_den = recall + precision

    f1 = f1_num / f1_den if f1_den > 0 else 0.

    return {"recall": recall, "precision": precision, "f1": f1}

def subtaskB(gold, submit, data, verbose=False):
    return match_relations(gold, submit, data)
