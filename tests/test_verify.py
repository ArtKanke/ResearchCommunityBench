"""Verification protocol against known-valid and known-invalid contributions on the real
materials environment (hidden quarter of Matbench expt_gap). Each case has a known answer:

  valid:   comparison that beats a mean predictor; preregistered equivalence of two RF seeds;
           an extension that materially uses another agent's table; a confirmed refutation
  invalid: fabricated number; a prediction that does not beat the baseline; a duplicate;
           superficial file reuse (reads the table, ignores it); a self-dependency; a citation
"""

import json

from tests.test_research import ADMIN, c, put, setup  # first: sets the test env

HEAD = """import json, os, sys
import numpy as np, pandas as pd
D, E, O, I = os.environ['DATA_DIR'], os.environ.get('EVAL_DIR'), os.environ['OUT_DIR'], os.environ.get('IN_DIR', '')
sys.path.insert(0, D + '/baselines')
from materials_baseline import featurize, parse_formula
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
tr = pd.read_csv(D + '/expt_gap/train.csv'); el = pd.read_csv(D + '/elements/elements.csv')
if E:
    trn, te = tr, pd.read_csv(E + '/expt_gap/inputs.csv')
else:
    trn, te = tr.iloc[:2700], tr.iloc[2700:]
def rf(seed):
    m = RandomForestRegressor(n_estimators=40, random_state=seed, n_jobs=-1).fit(featurize(trn.composition, el), trn.target)
    return m.predict(featurize(te.composition, el))
def mean():
    return np.full(len(te), trn.target.mean())
def effects_table(df):
    rows = {}
    for comp, y in zip(df.composition, df.target):
        for s in parse_formula(comp):
            rows.setdefault(s, []).append(y)
    return pd.DataFrame({'symbol': list(rows), 'effect': [np.mean(v) for v in rows.values()]})
def effect_feature(comps, table):
    eff = dict(zip(table.symbol, table.effect))
    out = []
    for comp in comps:
        amt = parse_formula(comp); w = sum(amt.values())
        out.append(sum(eff.get(s, np.nan) * n for s, n in amt.items()) / w)
    x = np.array(out); return np.where(np.isnan(x), np.nanmean(x), x)
def effect_model(table):
    m = LinearRegression().fit(effect_feature(trn.composition, table)[:, None], trn.target)
    return m.predict(effect_feature(te.composition, table)[:, None])
def emit(pa, pb=None, extra=None):
    if E:
        pd.DataFrame({'id': te.id, 'pred': pa}).to_csv(O + ('/predictions.csv' if pb is None else '/predictions_a.csv'), index=False)
        if pb is not None:
            pd.DataFrame({'id': te.id, 'pred': pb}).to_csv(O + '/predictions_b.csv', index=False)
    else:
        y = te.target.to_numpy()
        m = {'mae_a': round(float(np.mean(np.abs(pa - y))), 4)}
        if pb is not None:
            m['mae_b'] = round(float(np.mean(np.abs(pb - y))), 4)
        json.dump(m | (extra or {}), open(O + '/metrics.json', 'w'))
"""

CODE = {
    "rf_vs_mean": HEAD + "emit(rf(0), mean())\n",
    "mean_only": HEAD + "emit(mean())\n",
    "seed0_vs_seed1": HEAD + "emit(rf(0), rf(1))\n",
    "effects": HEAD + "t = effects_table(trn)\nif not E: t.to_csv(O + '/effects.csv', index=False)\nemit(effect_model(t), mean())\n",
    "uses_effects": HEAD + "t = pd.read_csv(I + '/effects.csv')\nemit(effect_model(t), mean())\n",
    "uses_effects_v2": HEAD + "t = pd.read_csv(I + '/effects.csv')\nemit(0.9 * effect_model(t) + 0.1 * trn.target.mean(), mean())\n",
    "ignores_effects": HEAD + "t = pd.read_csv(I + '/effects.csv')\nemit(rf(2), mean())\n",
}


def submit(h, kind, name, code, evidence, quantities, profile="comparison", relations=(), inputs=None, **extra):
    test = {"profile": profile, "dataset": "expt_gap", "code": code, "cmd": "python run.py", "inputs": inputs or {}}
    test |= {k: v for k, v in extra.items() if k in ("bounds", "better")}
    body = {"kind": kind, "title": name, "claim": {"statement": f"{name}: tested on the expt_gap dataset.", "datasets": ["expt_gap"],
            "quantities": quantities}, "evidence": evidence, "test": test, "relations": list(relations)}
    if "prespec" in extra:
        body["prespec"] = extra["prespec"]
    r = c.post("/artifacts", json=body, headers=h)
    assert r.status_code == 200, r.text
    return r.json()["artifact_id"]


def ex(h, name, inputs=None):
    code = put(h, {"run.py": CODE[name]})
    r = c.post("/exec", json={"code": code, "cmd": "python run.py", "inputs": inputs or {}}, headers=h).json()
    assert r["exit_code"] == 0, r["stdout"]
    return r


def q(name, v):
    return [{"name": name, "value": v}]


def test_verification_protocol_on_known_cases():
    rid, hf, hb = setup(domain="materials", minutes=30)
    hc = {**hf, "X-Agent": c.post("/agents/spawn", json={"role": "c", "brief": ""}, headers=hf).json()["agent_id"]}
    hd = {**hf, "X-Agent": c.post("/agents/spawn", json={"role": "d", "brief": ""}, headers=hf).json()["agent_id"]}

    e1 = ex(hf, "rf_vs_mean")
    good = submit(hf, "finding", "RF beats the mean predictor", e1["code"], [e1["exec_id"]], q("mae_a", e1["metrics"]["mae_a"]))
    fake = submit(hf, "finding", "Fabricated error", e1["code"], [e1["exec_id"]], q("mae_a", 0.05))
    e2 = ex(hf, "mean_only")
    weak = submit(hf, "finding", "Mean predictor is good enough", e2["code"], [e2["exec_id"]], q("mae_a", e2["metrics"]["mae_a"]), profile="prediction")
    dup = submit(hb, "finding", "Same RF result again", e1["code"], [e1["exec_id"]], q("mae_a", e1["metrics"]["mae_a"]))

    pre = c.post("/artifacts", json={"kind": "preregistration", "title": "Seed equivalence plan",
                                     "test": {"profile": "equivalence", "dataset": "expt_gap", "code": e1["code"], "cmd": "python run.py",
                                              "bounds": [-0.05, 0.05]}}, headers=hb).json()["artifact_id"]
    e3 = ex(hb, "seed0_vs_seed1")
    null = submit(hb, "null_result", "RF seed does not change error by more than 0.05 eV", e3["code"], [e3["exec_id"]],
                  q("mae_a", e3["metrics"]["mae_a"]), profile="equivalence", bounds=[-0.05, 0.05], prespec=pre)

    e4 = ex(hf, "effects")
    table = e4["outputs"]["effects.csv"]
    method = submit(hf, "method", "Element-effect table as a one-feature model", e4["code"], [e4["exec_id"]], q("mae_a", e4["metrics"]["mae_a"]))
    use = {"target": method, "relation": "extends", "used": [table]}
    e5 = ex(hb, "uses_effects", {"effects.csv": table})
    ext = submit(hb, "finding", "Effect table transfers as a feature", e5["code"], [e5["exec_id"]], q("mae_a", e5["metrics"]["mae_a"]),
                 relations=[use], inputs={"effects.csv": table})
    e6 = ex(hc, "ignores_effects", {"effects.csv": table})
    fake_collab = submit(hc, "finding", "Builds on the effect table (it does not)", e6["code"], [e6["exec_id"]], q("mae_a", e6["metrics"]["mae_a"]),
                         relations=[use], inputs={"effects.csv": table})
    e7 = ex(hf, "uses_effects_v2", {"effects.csv": table})
    self_ext = submit(hf, "finding", "Founder extends its own table", e7["code"], [e7["exec_id"]], q("mae_a", e7["metrics"]["mae_a"]),
                      relations=[use], inputs={"effects.csv": table})
    cite = submit(hd, "finding", "RF again, citing the method", put(hd, {"run.py": CODE["rf_vs_mean"].replace("rf(0)", "rf(5)")}),
                  [ex(hd, "rf_vs_mean")["exec_id"]], [], relations=[{"target": method, "relation": "cites"}])
    code8 = put(hd, {"run.py": CODE["rf_vs_mean"].replace("rf(0)", "rf(7)")})
    e8 = c.post("/exec", json={"code": code8, "cmd": "python run.py"}, headers=hd).json()
    refute = submit(hd, "finding", "The mean predictor is not good enough", e8["code"], [e8["exec_id"]], q("mae_a", e8["metrics"]["mae_a"]),
                    relations=[{"target": weak, "relation": "refutes"}])

    res = c.post(f"/admin/runs/{rid}/verify", headers=ADMIN).json()
    by = {a["id"]: a for a in res["artifacts"]}
    st = {k: by[v]["status"] for k, v in dict(good=good, fake=fake, weak=weak, dup=dup, null=null, method=method, ext=ext,
                                                fake_collab=fake_collab, self_ext=self_ext, cite=cite, refute=refute).items()}
    print(json.dumps(st, indent=1), json.dumps(res["summary"], indent=1))
    assert st["good"] == "accepted" and by[good]["checks"]["re_execution"] is True
    assert st["fake"] == "rejected" and by[fake]["fabrication"]
    assert st["weak"] == "rejected" and by[weak]["checks"]["hidden_replication"] is False
    assert st["dup"] == "duplicate"
    assert st["null"] == "accepted", by[null]
    assert st["method"] == "accepted" and st["ext"] == "accepted"
    rel = {(a["id"], v["relation"]): v for a in res["artifacts"] for v in a.get("relations", [])}
    assert rel[(ext, "extends")]["verified"] and rel[(ext, "extends")]["cross_agent"]          # material, cross-agent
    assert not rel[(fake_collab, "extends")]["verified"]                                      # superficial reuse
    assert rel[(self_ext, "extends")]["verified"] and not rel[(self_ext, "extends")]["cross_agent"]  # material but own work
    assert not rel[(cite, "cites")]["verified"]
    assert rel[(refute, "refutes")]["verified"] and st["refute"] == "accepted"
    s = res["summary"]
    assert s["C_collaborative"] == 2  # ext (extends another agent) and refute (refutes another agent); nothing else
    assert s["rejected"]["fabrication"] == 1 and s["duplicates"] == 1
    # frontier: only the first RF win advances expt_gap; later RF seeds and the weaker effect-table models do not
    assert by[good]["frontier"] and not by[cite]["frontier"] and not by[null]["frontier"]
    assert s["R_frontier"] == 1, s["R_frontier"]


def test_frontier_counts_only_improvements():
    from lab.verify import best_gain, mark_frontier
    arts, out = {}, {}
    cases = [("x", "prediction", .50, .9, "accepted"), ("x", "prediction", .495, .9, "accepted"),  # near-copy
             ("x", "comparison", .40, .45, "accepted"), ("y", "prediction", .9, .95, "accepted"),
             ("x", "comparison", .45, .6, "accepted"), ("x", "equivalence", .1, .1, "accepted"),
             ("x", "prediction", .1, .9, "rejected")]
    for i, (ds, prof, a, b, status) in enumerate(cases):
        arts[f"c{i}"] = {"test": {"dataset": ds, "profile": prof}}
        out[f"c{i}"] = {"id": f"c{i}", "status": status, "hidden": {"a": a, "b": b}}
    mark_frontier(arts, out)
    assert [out[f"c{i}"].get("frontier") for i in range(7)] == [True, False, True, True, False, False, None]
    acc = [r for r in out.values() if r["status"] == "accepted"]
    assert best_gain(acc, arts, {"x": 1.0, "y": 1.0}.get) == {"x": .6, "y": .1}
