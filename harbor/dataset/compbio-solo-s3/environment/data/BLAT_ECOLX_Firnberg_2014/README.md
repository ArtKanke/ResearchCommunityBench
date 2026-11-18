# BLAT_ECOLX_Firnberg_2014

Deep mutational scan: A Comprehensive, High-Resolution Map of a Gene's Fitness Landscape (Firnberg, 2014). Protein BLAT_ECOLX (Escherichia coli), 286 residues; phenotype: Growth (0.25-1024 ug/mL ampicillin) doubling (OrganismalFitness). From ProteinGym v1.3 (Notin et al., NeurIPS 2023; MIT).

- train.csv: 3659 single substitutions: id, mutant (e.g. A102E), sequence (full mutated sequence), target (DMS score; higher = fitter)
- A further 1124 variants are held by the evaluator.
- Wild-type sequence: MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW
- Evaluated by mean absolute error on the DMS score scale.
