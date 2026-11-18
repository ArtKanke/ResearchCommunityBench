"""Post-run verification of an open-ended research run. Nothing here is visible to agents.

For every artifact the agents submitted, in order:
  1. live checks (schema, provenance, prespecification) from submission time; a claimed number
     that no recorded run produced is the only thing classed as fabrication
  2. re-execution: the first evidence exec is rerun in a clean container; recorded metrics
     must agree within 10%
  3. the verification profile, run by the evaluator:
       prediction / comparison / equivalence: the artifact's test runs with the hidden inputs
       mounted at $EVAL_DIR; the server reads the predictions and scores them against hidden
       labels itself (never the agent's own metric), with a paired bootstrap
       custom: no hidden test exists -> tier "auditor-only", pending the jury
  4. dedup: identical hidden-data predictions from different artifacts are one contribution
  5. relations: extends must be material under an evaluator-controlled ablation (the consumed
     input is row-shuffled and the test rerun); corrects/refutes are confirmed when the target
     fails verification and the challenger passes; replicates strengthen the target, no credit
Then R (research output) and C (collaboration) counts, by evidence tier.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from . import executor
from .artifacts import CREDITED, Artifact, check_live, numbers, value_matches
from .research import domain_dir, run_artifacts, run_execs
from .server import q, q1

BOOT = 2000
REEXEC_TOL = 0.10
# Minimum meaningful effect (open decision D-E4): an improvement must exceed this share of the comparator's error,
# not merely have a CI that excludes zero (E4 showed 0.002 eV gains pass otherwise).
MIN_EFFECT_REL = float(os.environ.get("LAB_MIN_EFFECT_REL", "0.02"))
WORKERS = int(os.environ.get("LAB_VERIFY_WORKERS", "6"))  # artifacts verified in parallel (GPU domains: fewer)
TEST_TIMEOUT = int(os.environ.get("LAB_TEST_TIMEOUT", "900"))  # seconds per hidden-data test


# ---------------------------------------------------------------- statistics

def losses(pred: pd.DataFrame, labels: pd.DataFrame, task: str) -> tuple[np.ndarray, float]:
    """Per-sample loss on the hidden set, in the labels' order, and coverage of hidden ids."""
    p = pred.drop_duplicates("id").set_index("id")["pred"]
    m = labels.set_index("id")["target"]
    cover = float(m.index.isin(p.index).mean())
    pv = p.reindex(m.index).to_numpy(float)
    y = m.to_numpy(float)
    if task == "classification":
        multiclass = len(np.unique(y[~np.isnan(y)])) > 2
        loss = (np.rint(pv) != y).astype(float) if multiclass else ((pv > .5) != (y > .5)).astype(float)
    else:
        loss = np.abs(pv - y)
    # a missing or NaN prediction is never free: classification counts it wrong, regression scores it as the
    # mean predictor would (so writing NaN where a model would lose gains nothing)
    missing = np.isnan(pv) & ~np.isnan(y)
    if task == "classification":
        loss[missing] = 1.0
    else:
        loss[missing] = np.abs(y[missing] - np.nanmean(y))
    return loss, cover


def paired_ci(diff: np.ndarray, level: float = .95, seed: int = 0) -> tuple[float, float, float]:
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (BOOT, len(diff)))
    means = diff[idx].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(diff.mean()), float(lo), float(hi)


@lru_cache(maxsize=16)
def baseline_losses(domain_root: str, dataset: str) -> np.ndarray | float:
    """The `prediction` comparator: per-sample losses of the environment's baseline (hidden/<ds>/baseline.csv),
    a fixed published reference error (reference.json), or, for materials, the shipped baseline model."""
    hid_ds = os.path.join(domain_root, "hidden", dataset)
    meta = json.load(open(os.path.join(hid_ds, "meta.json")))
    if os.path.exists(os.path.join(hid_ds, "reference.json")):
        return float(json.load(open(os.path.join(hid_ds, "reference.json")))["error"])
    if os.path.exists(os.path.join(hid_ds, "baseline.csv")):
        return losses(pd.read_csv(os.path.join(hid_ds, "baseline.csv")), pd.read_csv(os.path.join(hid_ds, "labels.csv")), meta["task"])[0]
    from .materials import baseline_predict

    pub, hid = os.path.join(domain_root, "public"), os.path.join(domain_root, "hidden")
    meta = json.load(open(os.path.join(hid, dataset, "meta.json")))
    tr = pd.read_csv(os.path.join(pub, dataset, "train.csv"))
    inp = pd.read_csv(os.path.join(hid, dataset, "inputs.csv"))
    el = pd.read_csv(os.path.join(pub, "elements", "elements.csv"))
    pred = pd.DataFrame({"id": inp["id"], "pred": baseline_predict(tr, inp, el, meta["task"])})
    return losses(pred, pd.read_csv(os.path.join(hid, dataset, "labels.csv")), meta["task"])[0]


# ---------------------------------------------------------------- one artifact

def _read_csv(blob):
    df = pd.read_csv(io.BytesIO(executor.get_blob(blob)))
    if "id" not in df.columns or "pred" not in df.columns:
        raise ValueError("predictions need columns id, pred")
    return df


def run_test(art, run, input_files=None):
    t = art["test"]
    return executor.run(t["code"], t["cmd"], t.get("inputs", {}), domain_dir(run, "public"), timeout=TEST_TIMEOUT,
                        eval_dir=domain_dir(run, "eval"), input_files=input_files)


def score_test(art, run, rec) -> dict:
    """Server-side scoring of a test exec's predictions against hidden labels."""
    t = art["test"]
    ds, prof = t["dataset"], t["profile"]
    hid = domain_dir(run, "hidden")
    if not hid or not os.path.isdir(os.path.join(hid, ds)):
        return {"pass": False, "reason": f"unknown dataset {ds}"}
    meta = json.load(open(os.path.join(hid, ds, "meta.json")))
    labels = pd.read_csv(os.path.join(hid, ds, "labels.csv"))
    out = rec["outputs"]
    if rec["exit_code"] != 0:
        return {"pass": False, "reason": f"test crashed on hidden inputs (exit {rec['exit_code']})", "stdout": rec["stdout"][-500:]}
    try:
        if prof == "prediction":
            la, ca = losses(_read_csv(out["predictions.csv"]), labels, meta["task"])
            lb, cb = baseline_losses(os.path.dirname(hid), ds), 1.0
        else:
            la, ca = losses(_read_csv(out["predictions_a.csv"]), labels, meta["task"])
            lb, cb = losses(_read_csv(out["predictions_b.csv"]), labels, meta["task"])
    except (KeyError, ValueError) as e:
        return {"pass": False, "reason": f"test did not write the profile's prediction files ({e})"}
    pred_a = _read_csv(out["predictions.csv" if prof == "prediction" else "predictions_a.csv"]).drop_duplicates("id").set_index("id")["pred"]
    pred_a = pred_a.reindex(labels["id"]).to_numpy(float)
    pred_b = None
    if prof != "prediction" and "predictions_b.csv" in out:
        pred_b = _read_csv(out["predictions_b.csv"]).drop_duplicates("id").set_index("id")["pred"].reindex(labels["id"]).to_numpy(float)
    if min(ca, cb) < .95:
        return {"pass": False, "reason": f"predictions cover only {min(ca, cb):.0%} of hidden inputs"}
    diff = la - lb
    res = {"metric": meta["metric"], "n": int(np.sum(~np.isnan(diff))), "a": float(np.nanmean(la)), "b": float(np.nanmean(lb)),
           "losses": la, "pred": (prof, t.get("better", "a"), pred_a, pred_b)}  # kept in memory for ablations and dedup, stripped before storing
    if prof == "equivalence":
        mean, lo, hi = paired_ci(diff, .90)
        lo_b, hi_b = t["bounds"]
        res |= {"diff": mean, "ci90": [lo, hi], "bounds": [lo_b, hi_b], "pass": lo > lo_b and hi < hi_b}
        res["reason"] = (f"TOST: 90% CI of {meta['metric']} difference [{lo:.4f}, {hi:.4f}] inside [{lo_b}, {hi_b}]" if res["pass"]
                         else f"not shown equivalent: 90% CI [{lo:.4f}, {hi:.4f}] not inside [{lo_b}, {hi_b}]")
        return res
    mean, lo, hi = paired_ci(diff if t.get("better", "a") == "a" or prof == "prediction" else -diff)
    better = "baseline" if prof == "prediction" else ("b" if t.get("better") == "b" else "a")
    margin = MIN_EFFECT_REL * float(np.nanmean(lb if prof == "prediction" or t.get("better", "a") == "a" else la))
    res |= {"diff": mean, "ci95": [lo, hi], "min_effect": margin, "pass": hi < -margin}
    who = "model" if prof == "prediction" else f"arm {better}"
    vs = "the environment baseline" if prof == "prediction" else "the other arm"
    res["reason"] = (f"hidden {meta['metric']}: {who} beats {vs} (difference {mean:+.4f}, 95% CI [{lo:.4f}, {hi:.4f}], "
                     f"beyond the minimum effect {margin:.4f})" if res["pass"]
                     else f"did not replicate on hidden data by the minimum effect: difference {mean:+.4f}, 95% CI "
                          f"[{lo:.4f}, {hi:.4f}], needed below {-margin:.4f}")
    return res


def same_claim(p, q) -> bool:
    """Same profile, same direction, numerically the same predictions in every arm."""
    if p is None or q is None or p[:2] != q[:2]:
        return False
    for x, y in zip(p[2:], q[2:]):
        if (x is None) != (y is None):
            return False
        if x is not None and not (x.shape == y.shape and np.allclose(x, y, rtol=1e-6, atol=1e-9, equal_nan=True)):
            return False
    return True


_reruns: dict = {}
_rerun_lock = __import__("threading").Lock()


def _rerun(e, run):
    """Each evidence exec is rerun once per verification, however many artifacts cite it."""
    with _rerun_lock:
        ev = _reruns.setdefault(e["id"], __import__("threading").Event())
        first = not hasattr(ev, "rec")
        if first:
            ev.rec = None
    if first:
        ev.rec = executor.run(e["code"], e["cmd"], e.get("inputs", {}), domain_dir(run, "public"), timeout=max(TEST_TIMEOUT, 900))
        ev.set()
    else:
        ev.wait()
    return ev.rec


def reexecute(art, execs, run) -> dict:
    """Rerun the first evidence exec; the numbers the claim states must reproduce within 10%
    (other numbers in metrics.json, such as timings, are not compared)."""
    ev = [execs[e] for e in art.get("evidence", []) if e in execs]
    if not ev:
        return {"pass": None, "reason": "no evidence exec to rerun"}
    e = ev[0]
    rec = _rerun(e, run)
    if rec["exit_code"] != 0:
        return {"pass": False, "reason": f"evidence exec {e['id']} fails on rerun (exit {rec['exit_code']})", "seconds": rec["seconds"]}
    claimed = [q["value"] for q in (art.get("claim") or {}).get("quantities", [])
               if any(value_matches(q["value"], n) for n in numbers(e.get("metrics")))]
    if claimed:
        fresh = list(numbers(rec.get("metrics")))
        ok = all(any(abs(v - y) <= REEXEC_TOL * max(abs(v), 1e-9) or abs(v - y) < 1e-6 for y in fresh) for v in claimed)
        return {"pass": ok, "seconds": rec["seconds"],
                "reason": "claimed numbers reproduce within 10%" if ok else "claimed numbers do not reproduce within 10% on rerun"}
    return {"pass": True, "seconds": rec["seconds"], "reason": "evidence reruns without error (no claimed numbers to compare)"}


def ablate(art, run, base: dict, rel: dict) -> dict:
    """Evaluator-controlled intervention: row-shuffle every consumed CSV from the target, rerun
    the test on hidden data, and ask whether the result gets materially worse."""
    t = art["test"]
    files = {}
    for name, blob in t.get("inputs", {}).items():
        if blob in rel.get("used", []):
            try:
                df = pd.read_csv(io.BytesIO(executor.get_blob(blob)))
            except Exception:
                continue
            # lookup keys (ids, mutants, sequences, compositions, symbols) are the non-numeric columns;
            # shuffling the numeric columns against them breaks what the table says about each key
            keys = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) or c.lower() in ("id", "position", "pos")]
            other = [c for c in df.columns if c not in keys]
            if not other:
                continue
            perm = np.random.default_rng(1).permutation(len(df))
            df[other] = df[other].to_numpy()[perm]
            files[name] = df.to_csv(index=False).encode()
    if not files:
        return {"material": None, "reason": "no consumed table to intervene on (code reuse or non-tabular input): dependency not ablated"}
    rec = run_test(art, run, input_files=files)
    s = score_test(art, run, rec)
    if "losses" not in s or "losses" not in base:  # a crash is not evidence of use (code can simply require the file)
        return {"material": False, "reason": "test fails without the consumed data: inconclusive, not counted as material use"}
    mean, lo, hi = paired_ci(s["losses"] - base["losses"])
    margin = MIN_EFFECT_REL * float(np.nanmean(base["losses"]))
    ok = lo > margin
    return {"material": ok, "effect": mean, "ci95": [lo, hi], "min_effect": margin,
            "reason": (f"shuffling the consumed data worsens hidden error by {mean:+.4f} (95% CI [{lo:.4f}, {hi:.4f}], "
                       f"beyond the minimum effect {margin:.4f})" if ok
                       else f"result does not materially depend on the consumed data (effect {mean:+.4f}, 95% CI [{lo:.4f}, {hi:.4f}])")}


# ---------------------------------------------------------------- the run

def verify_run(rid: str) -> dict:
    _reruns.clear()
    run = dict(q1("select * from runs where id=?", rid))
    arts, execs = run_artifacts(rid), run_execs(rid)
    t_start = time.time()
    out, losses_by, fp = {}, {}, {}

    def check(aid):
        """One artifact's verification; an error in one artifact never stops the others."""
        try:
            return _check(aid)
        except KeyError as e:
            a = arts[aid]
            return ({"id": aid, "kind": a["kind"], "agent": a["agent"], "title": a["title"], "tier": None, "checks": {},
                     "fabrication": False, "status": "rejected", "reason": f"verification failed: a referenced file (blob {str(e)[:20]}) was never uploaded"}, None, None)
        except Exception as e:  # noqa: BLE001
            a = arts[aid]
            return ({"id": aid, "kind": a["kind"], "agent": a["agent"], "title": a["title"], "tier": None, "checks": {},
                     "fabrication": False, "status": "rejected", "reason": f"verification error: {type(e).__name__}: {str(e)[:200]}"}, None, None)

    def _check(aid):
        """Everything expensive for one artifact (independent of the others): live checks, rerun, hidden test."""
        a = arts[aid]
        before = {k: v for k, v in arts.items() if v["ts"] < a["ts"]}
        body = {k: a[k] for k in Artifact.model_fields if k in a}
        a["live"] = check_live(Artifact.model_validate(body), execs, before, a["ts"])  # current rules, not submission-time
        r = {"id": aid, "kind": a["kind"], "agent": a["agent"], "title": a["title"], "tier": None,
             "checks": dict(a["live"]["checks"]), "fabrication": a["live"].get("fabrication", False)}
        if a["kind"] not in CREDITED:
            return r | {"status": "not credited", "reason": f"{a['kind']}s are never credited"}, None, None
        if not a["live"]["ok"]:
            return r | {"status": "rejected", "reason": "; ".join(a["live"]["why"])}, None, None
        if a["kind"] != "hypothesis":
            rx = reexecute(a, execs, run)
            r["checks"]["re_execution"] = rx["pass"]
            r["reexec"] = rx
            if rx["pass"] is False:
                return r | {"status": "rejected", "reason": rx["reason"]}, None, None
        if a["test"]["profile"] == "custom":
            return r | {"status": "pending auditor", "tier": "auditor-only",
                        "reason": "custom evidence: re-executed, no hidden-data test; awaits the calibrated jury"}, None, None
        s = score_test(a, run, run_test(a, run))
        loss, pred = s.pop("losses", None), s.pop("pred", None)
        r["hidden"] = s
        r["checks"]["hidden_replication"] = bool(s.get("pass"))
        if not s.get("pass"):
            return r | {"status": "rejected", "reason": s["reason"]}, loss, None
        return r | {"status": "candidate", "reason": s["reason"]}, loss, pred

    # the expensive part runs in parallel; statuses are assigned in submission order (dedup keeps the earliest)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = dict(zip(arts, pool.map(check, list(arts))))
    for aid, a in arts.items():
        r, loss, pred = results[aid]
        out[aid] = r
        losses_by[aid] = loss
        if r["status"] != "candidate":
            continue
        ds = a["test"]["dataset"]
        dup = next((other for other, p in fp.get(ds, []) if same_claim(p, pred)), None)
        if dup:
            r |= {"status": "duplicate", "reason": f"same hidden-data predictions as {dup}", "duplicate_of": dup}
            continue
        fp.setdefault(ds, []).append((aid, pred))
        r |= {"status": "accepted", "tier": "replicated"}

    mark_frontier(arts, out)

    # relations, after every artifact has a status
    for aid, a in arts.items():
        r = out[aid]
        r["relations"] = []
        for rel in a.get("relations", []):
            tgt = out.get(rel["target"])
            v = {"target": rel["target"], "relation": rel["relation"], "cross_agent": bool(tgt) and tgt["agent"] != a["agent"]}
            if rel["relation"] == "cites" or not tgt:
                v |= {"verified": False, "reason": "citation: recorded, never credited"}
            elif rel["relation"] == "replicates":
                v["verified"] = r.get("status") == "accepted" and tgt.get("status") == "accepted"
                if v["verified"]:
                    tgt.setdefault("replicated_by", []).append(aid)
                v["reason"] = "independent replication strengthens the target (no new credit)" if v["verified"] else "replication not confirmed"
            elif rel["relation"] in ("corrects", "refutes"):
                v["verified"] = r.get("status") == "accepted" and tgt.get("status") in ("rejected",)
                v["reason"] = ("challenge confirmed: the target fails verification, the challenger passes" if v["verified"]
                               else "challenge not confirmed by the evaluator")
                if v["verified"]:
                    tgt.setdefault("refuted_by", []).append(aid)
            else:  # extends
                if r.get("status") != "accepted":
                    v |= {"verified": False, "reason": "the extension itself is not an accepted contribution"}
                elif tgt.get("status") != "accepted":
                    v |= {"verified": False, "reason": "the extended work is not an accepted contribution"}
                else:
                    ab = ablate(a, run, {"losses": losses_by.get(aid)} if losses_by.get(aid) is not None else {}, rel)
                    v |= {"verified": bool(ab["material"]), "ablation": ab, "reason": ab["reason"]}
            r["relations"].append(v)

    # scores
    acc = [r for r in out.values() if r.get("status") == "accepted"]
    collab = [r for r in acc if any(v["verified"] and v["cross_agent"] and v["relation"] in ("extends", "corrects", "refutes")
                                    for v in r["relations"])]
    credited = [r for r in out.values() if r["kind"] in CREDITED]
    summary = {
        "R_replicated": len(acc),
        "R_frontier": sum(bool(r.get("frontier")) for r in acc),
        # best accepted held-out gain over the environment baseline, per dataset the community advanced
        "best_gain": best_gain(acc, arts, _baseline_mean(run)),
        "R_auditor_only_pending": sum(r.get("status") == "pending auditor" for r in out.values()),
        "C_collaborative": len(collab),
        "submitted_credited": len(credited),
        "rejected": {k: sum(1 for r in credited if r.get("status") == "rejected" and cond(r)) for k, cond in {
            "fabrication": lambda r: r["fabrication"],
            "miscited_evidence": lambda r: not r["fabrication"] and "mis-cited evidence" in r.get("reason", ""),
            "failed_live_checks": lambda r: not r["fabrication"] and "mis-cited evidence" not in r.get("reason", "") and r["checks"] and not all(v for k, v in r["checks"].items() if k in ("evidence_exists", "evidence_succeeded", "quantities_recorded", "targets_exist", "used_inputs_consumed", "prespec_before_evidence")),
            "not_reproducible": lambda r: r["checks"].get("re_execution") is False,
            "not_replicated_on_hidden": lambda r: r["checks"].get("hidden_replication") is False,
        }.items()},
        "duplicates": sum(r.get("status") == "duplicate" for r in out.values()),
        "verification_seconds": round(time.time() - t_start, 1),
    }
    summary["failure_modes"] = failure_modes(rid, out, arts, execs)
    return {"run_id": rid, "domain": run.get("domain"), "summary": summary, "artifacts": list(out.values())}


def winning_loss(art, r) -> float | None:
    """Held-out loss of the arm a prediction or comparison claims is better; None for other test types."""
    t, h = art["test"], r.get("hidden") or {}
    if t["profile"] == "prediction":
        return h.get("a")
    if t["profile"] == "comparison":
        return h.get("b") if t.get("better") == "b" else h.get("a")
    return None


def _baseline_mean(run):
    root = os.path.dirname(domain_dir(run, "hidden") or "")
    cache: dict[str, float | None] = {}

    def mean(ds):
        if ds not in cache:
            try:
                cache[ds] = float(np.nanmean(baseline_losses(root, ds)))
            except Exception:  # noqa: BLE001
                cache[ds] = None
        return cache[ds]
    return mean


def mark_frontier(arts, out) -> None:
    """An accepted prediction or comparison counts toward the frontier only if its winning arm beats the best earlier
    frontier result on the same dataset by the minimum effect, so near-copies of one win add no credit. Equivalence
    and null results never advance the frontier (they still count in R_replicated)."""
    best: dict[str, float] = {}
    for aid, a in arts.items():
        r = out[aid]
        if r.get("status") != "accepted":
            continue
        loss = winning_loss(a, r)
        if loss is None:
            r["frontier"] = False
            continue
        ds = a["test"]["dataset"]
        r["frontier"] = ds not in best or loss < best[ds] * (1 - MIN_EFFECT_REL)
        if r["frontier"]:
            best[ds] = loss


def best_gain(acc, arts, baseline) -> dict[str, float]:
    """Per dataset: relative held-out loss reduction of the best accepted result over the environment baseline
    (`baseline(ds)` gives its mean loss, or None when unavailable)."""
    best: dict[str, float] = {}
    for r in acc:
        loss = winning_loss(arts[r["id"]], r)
        if loss is not None:
            ds = arts[r["id"]]["test"]["dataset"]
            best[ds] = min(best.get(ds, loss), loss)
    out = {}
    for ds, loss in best.items():
        b = baseline(ds)
        if b:
            out[ds] = round(1 - loss / b, 4)
    return out


def failure_modes(rid, out, arts, execs) -> dict:
    """Diagnostics for the ways research communities fail (reported, not scored)."""
    acc = [r for r in out.values() if r.get("status") == "accepted"]
    credited = [r for r in out.values() if r["kind"] in CREDITED]
    agents = {r["agent"] for r in out.values()} | {e["agent"] for e in execs.values()}
    calls = q1("select count(*) n from events where run=? and type='complete'", rid)["n"]
    msgs = q1("select count(*) n from events where run=? and type='post'", rid)["n"]
    cost = q1("select coalesce(sum(cost),0) c from agents where run=?", rid)["c"]
    consumed = set()  # accepted work that another agent's experiment actually consumed
    for e in execs.values():
        for a in acc:
            outs = set()
            for eid in arts[a["id"]].get("evidence", []):
                if eid in execs:
                    outs |= set(execs[eid].get("outputs", {}).values())
            if outs & set(e.get("inputs", {}).values()) and e["agent"] != a["agent"]:
                consumed.add(a["id"])
    built_on_invalid = sum(1 for r in out.values() for v in r.get("relations", [])
                           if v["relation"] in ("extends", "replicates") and out.get(v["target"], {}).get("status") in ("rejected", "duplicate"))
    endorsed_invalid = sum(1 for r in out.values() for v in r.get("relations", [])
                           if v["relation"] == "replicates" and out.get(v["target"], {}).get("status") == "rejected")
    caught = sum(1 for r in out.values() for v in r.get("relations", []) if v["relation"] in ("corrects", "refutes") and v.get("verified"))
    datasets = [arts[r["id"]].get("test", {}).get("dataset") for r in credited if arts[r["id"]].get("test")]
    shares = np.array([datasets.count(d) for d in set(datasets)], float)
    entropy = float(-(shares / shares.sum() * np.log(shares / shares.sum())).sum() / np.log(len(shares))) if len(shares) > 1 else 0.0
    per = lambda x: round(x / len(acc), 3) if acc else None
    return {
        "coordination_overhead": {"cost_per_accepted": per(cost), "calls_per_accepted": per(calls), "messages_per_accepted": per(msgs),
                                  "agents": len(agents)},
        "knowledge_transfer": {"accepted_consumed_by_another_agent": len(consumed), "share": per(len(consumed)) if acc else None,
                               "duplicates": sum(r.get("status") == "duplicate" for r in out.values())},
        "invalid_propagation": {"contributions_built_on_failed_work": built_on_invalid},
        "allocation": {"rejected_share": round(sum(r.get("status") == "rejected" for r in credited) / len(credited), 3) if credited else None,
                       "dataset_entropy": round(entropy, 3), "datasets_touched": len(set(datasets))},
        "peer_review": {"replications_of_failed_work": endorsed_invalid, "confirmed_corrections_or_refutations": caught},
    }


def stable(obj):
    """Hash for change detection of stored verification results."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]
