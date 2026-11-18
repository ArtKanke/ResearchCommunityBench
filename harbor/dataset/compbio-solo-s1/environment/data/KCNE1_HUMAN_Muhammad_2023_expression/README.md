# KCNE1_HUMAN_Muhammad_2023_expression

Deep mutational scan: High-throughput functional mapping of variants in an arrhythmia gene, KCNE1, reveals novel biology (Muhammad, 2023). Protein KCNE1_HUMAN (Homo sapiens), 129 residues; phenotype: cell surface expression (Expression). From ProteinGym v1.3 (Notin et al., NeurIPS 2023; MIT).

- train.csv: 1757 single substitutions: id, mutant (e.g. A102E), sequence (full mutated sequence), target (DMS score; higher = fitter)
- A further 582 variants are held by the evaluator.
- Wild-type sequence: MILSNTTAVTPFLTKLWQETVQQGGNMSGLARRSPRSGDGKLEALYVLMVLGFFGFFTLGIMLSYIRSKKLEHSNDPFNVYIESDAWQEKDKAYVQARVLESYRSCYVVENHLAIEQPNTHLPETKPSP
- Evaluated by mean absolute error on the DMS score scale.
