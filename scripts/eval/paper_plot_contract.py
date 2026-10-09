"""Required identities for the clean paper bundle, including writeup and Pareto."""
REQUIRED_FIGURES = frozenset((group, name) for group, names in (
    ('01_quality', ('architecture_quality', 'quality_matrices', 'final_letter_comparison')),
    ('02_execution', ('first_error_survival', 'first_error_hazard', 'first_error_taxonomy')),
    ('03_structure', ('cycle_strata', 'final_strata', 'final_failed_trajectories')),
    ('04_stopping', ('requested_actual_stops', 'timing_failure_modes', 'execution_timing_decomposition', 'stop_residuals_and_coincidences')),
    ('05_learning', ('executor_validation_learning', 'executor_population_progression', 'controller_development')),
    ('06_writeup', ('baseline', 'matched_baseline', 'attempt_01', 'attempt_02', 'attempt_03', 'attempt_04', 'attempt_05')),
    ('07_compute', ('pareto_joint_success', 'pareto_strict_success')),
) for name in names)


def validate_manifest(manifest: dict) -> None:
    """Prevent old-export retirement if any required family/format is absent."""
    entries = manifest.get('figures', [])
    identities = [(e.get('group'), e.get('name')) for e in entries]
    if manifest.get('status') != 'complete' or set(identities) != REQUIRED_FIGURES or len(identities) != len(REQUIRED_FIGURES):
        raise ValueError('Replacement figure set incomplete or contains duplicate identities')
    if any(set(e.get('exports', {})) != {'png', 'pdf', 'svg'} for e in entries):
        raise ValueError('Replacement PNG/PDF/SVG formats incomplete')
