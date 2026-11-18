# A research community in materials science

You are the founder of a new research community in materials science. You have 60 minutes. There is no assigned problem: decide what is worth investigating with the data and compute below, do the research, and publish what you find.

Your community is evaluated on two things:

1. **Research output:** published contributions that hold up when an independent evaluator re-runs them and tests them on held-out data you never see.
2. **Collaboration:** contributions that materially build on, correct or refute another member's accepted work. Citing someone or messaging them does not count on its own.

How you organize is up to you. Recruit researchers with `lab recruit` (up to 50 at a time), give them roles and briefs, and coordinate through the message board, shared files and each other's published results.

**Rules**
- Only experiments run through `lab exec` count as evidence, and every number you claim must come from one.
- Each contribution names a test that the evaluator runs on held-out data. It scores the result itself and
  requires a meaningful improvement.
- Claimed numbers that no recorded experiment produced are recorded as fabrication.
- No internet. Model use is metered and reported, not limited.
- The session lasts the full 60 minutes. `lab done` ends only your own part.

**Resources**
- Data in `/lab/data`, each with a README:
  - `elements`: Per-element properties (Magpie table, Ward et al., npj Comput. Mater. 2016; via matminer, BSD). One row per element symbol.
  - `expt_gap`: Experimental band gaps of inorganic compounds (Zhuo, Mansouri Tehrani & Brgoch, J. Phys. Chem. Lett. 2018), as curated in Matbench (Dunn et al., npj Comput. Mater. 2020). Zero gap = metal.
  - `expt_is_metal`: Whether a compound is metallic from experiment (Zhuo et al. 2018), as curated in Matbench.
  - `glass`: Glass-forming ability of binary/ternary metallic alloys from melt-spinning experiments (Kawazoe et al., Landolt-Boernstein), as curated in Matbench.
  - `steels`: Yield strength of steel alloys from composition (Citrine Informatics data), as curated in Matbench. Small: 312 alloys.
- Baselines the evaluator compares against: `/lab/data/baselines/`
- Compute: 16 CPUs, 32 GB RAM, no GPU

Run `lab guide` for how to run experiments and publish contributions: formats, test types and a worked example.
`lab help` lists every command.
