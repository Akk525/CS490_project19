from typing import Dict


COOKING_TEMPLATES: Dict[str, str] = {
    'missing_tool': 'You do not have the required tool. Adapt using a commonly available substitute.',
    'missing_ingredient': 'A key ingredient is missing. Adapt the recipe while preserving the intended dish.',
    'incorrect_object': 'An incorrect ingredient/object was used. Recover without restarting from scratch.',
    'step_failure': 'The step failed (e.g., burned/overmixed/undercooked). Recover with minimal waste.',
    'environmental_constraint': 'Constraint: limited heat source or time. Adapt the procedure accordingly.',
}

DIY_TEMPLATES: Dict[str, str] = {
    'missing_tool': 'The specified tool is unavailable. Use an alternative safe method.',
    'missing_ingredient': 'A required material is unavailable. Suggest an appropriate replacement.',
    'incorrect_object': 'The wrong part/material was applied. Provide a corrective adaptation.',
    'step_failure': 'A step failed during execution. Recover while keeping the main objective.',
    'environmental_constraint': 'Environmental limits (space/weather/noise) now apply. Adapt the plan safely.',
}
