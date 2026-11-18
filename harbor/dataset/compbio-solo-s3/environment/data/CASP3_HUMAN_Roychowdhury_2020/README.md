# CASP3_HUMAN_Roychowdhury_2020

Deep mutational scan: Microfluidic deep mutational scanning of the human executioner caspases reveals differences in structure and regulation (Roychowdhury, 2022). Protein CASP3_HUMAN (Homo sapiens), 258 residues; phenotype: Fluorescence measurement (Activity). From ProteinGym v1.3 (Notin et al., NeurIPS 2023; MIT).

- train.csv: 1164 single substitutions: id, mutant (e.g. A102E), sequence (full mutated sequence), target (DMS score; higher = fitter)
- A further 403 variants are held by the evaluator.
- Wild-type sequence: MSGISLDNSYKMDYPEMGLCIIINNKNFHKSTGMTSRSGTDVDAANLRETFRNLKYEVRNKNDLTREEIVELMRDVSKEDHSKRSSFVCVLLSHGEEGIIFGTNGPVDLKKITNFFRGDRCRSLTGKPKLFIIQACRGTELDCGIETDSGVDDDMACHKIPVEADFLYAYSTAPGYYSWRNSKDGSWFIQSLCAMLKQYADKLEFMHILTRVNRKVATEFESFSFDATFHAKKQIPCIVSMLTKELYFYHLEHHHHHH
- Evaluated by mean absolute error on the DMS score scale.
