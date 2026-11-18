# RASK_HUMAN_Weng_2022_abundance

Deep mutational scan: The energetic and allosteric landscape for KRAS inhibition (Weng, 2022). Protein RASK_HUMAN (Homo sapiens), 188 residues; phenotype: Yeast growth (Expression). From ProteinGym v1.3 (Notin et al., NeurIPS 2023; MIT).

- train.csv: 2322 single substitutions: id, mutant (e.g. A102E), sequence (full mutated sequence), target (DMS score; higher = fitter)
- A further 744 variants are held by the evaluator.
- Wild-type sequence: MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM
- Evaluated by mean absolute error on the DMS score scale.
