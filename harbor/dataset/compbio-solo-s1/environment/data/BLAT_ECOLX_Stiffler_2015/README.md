# BLAT_ECOLX_Stiffler_2015

Deep mutational scan: Evolvability as a Function of Purifying Selection in TEM-1 β-lactamase (Stiffler, 2015). Protein BLAT_ECOLX (Escherichia coli), 286 residues; phenotype: Growth (10-2500 ug/mL ampicillin) (OrganismalFitness). From ProteinGym v1.3 (Notin et al., NeurIPS 2023; MIT).

- train.csv: 3753 single substitutions: id, mutant (e.g. A102E), sequence (full mutated sequence), target (DMS score; higher = fitter)
- A further 1243 variants are held by the evaluator.
- Wild-type sequence: MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW
- Evaluated by mean absolute error on the DMS score scale.
