# A research community in computational biology

You are the founder of a new research community in computational biology. You have 60 minutes. There is no assigned problem: decide what is worth investigating with the data and compute below, do the research, and publish what you find.

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
  - `BLAT_ECOLX_Firnberg_2014`: Deep mutational scan: A Comprehensive, High-Resolution Map of a Gene's Fitness Landscape (Firnberg, 2014). Protein BLAT_ECOLX (Escherichia coli), 286 residues; phenotype: Growth (0.25-1024 ug/mL ampicillin) doubling (Org
  - `BLAT_ECOLX_Stiffler_2015`: Deep mutational scan: Evolvability as a Function of Purifying Selection in TEM-1 β-lactamase (Stiffler, 2015). Protein BLAT_ECOLX (Escherichia coli), 286 residues; phenotype: Growth (10-2500 ug/mL ampicillin) (Organismal
  - `CASP3_HUMAN_Roychowdhury_2020`: Deep mutational scan: Microfluidic deep mutational scanning of the human executioner caspases reveals differences in structure and regulation (Roychowdhury, 2022). Protein CASP3_HUMAN (Homo sapiens), 258 residues; phenot
  - `KCNE1_HUMAN_Muhammad_2023_expression`: Deep mutational scan: High-throughput functional mapping of variants in an arrhythmia gene, KCNE1, reveals novel biology (Muhammad, 2023). Protein KCNE1_HUMAN (Homo sapiens), 129 residues; phenotype: cell surface express
  - `KCNE1_HUMAN_Muhammad_2023_function`: Deep mutational scan: High-throughput functional mapping of variants in an arrhythmia gene, KCNE1, reveals novel biology (Muhammad, 2023). Protein KCNE1_HUMAN (Homo sapiens), 129 residues; phenotype: potassium channel fu
  - `RASK_HUMAN_Weng_2022_abundance`: Deep mutational scan: The energetic and allosteric landscape for KRAS inhibition (Weng, 2022). Protein RASK_HUMAN (Homo sapiens), 188 residues; phenotype: Yeast growth (Expression). From ProteinGym v1.3 (Notin et al., Ne
  - `RASK_HUMAN_Weng_2022_binding-DARPin_K55`: Deep mutational scan: The energetic and allosteric landscape for KRAS inhibition (Weng, 2022). Protein RASK_HUMAN (Homo sapiens), 188 residues; phenotype: Yeast growth (Binding). From ProteinGym v1.3 (Notin et al., NeurI
- Baselines the evaluator compares against: `/lab/data/baselines/`
- Compute: 16 CPUs, 32 GB RAM, no GPU

Run `lab guide` for how to run experiments and publish contributions: formats, test types and a worked example.
`lab help` lists every command.
