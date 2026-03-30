import pandas as pd


def hardest_examples(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    score_cols = ['feasibility_score', 'adaptation_quality_score', 'helpfulness_score']
    df = df.copy()
    df['composite'] = df[score_cols].mean(axis=1)
    return df.sort_values('composite', ascending=True).head(n)


def add_failure_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'constraint_violation' not in df.columns:
        df['constraint_violation'] = 0.0
    if 'feasibility_score' not in df.columns:
        df['feasibility_score'] = 0.0
    if 'adaptation_quality_score' not in df.columns:
        df['adaptation_quality_score'] = 0.0
    if 'helpfulness_score' not in df.columns:
        df['helpfulness_score'] = 0.0
    if 'semantic_similarity_score' not in df.columns:
        df['semantic_similarity_score'] = 0.0

    def classify(row) -> str:
        if float(row['constraint_violation']) >= 0.7:
            return 'constraint_failure'
        if float(row['feasibility_score']) < 0.35:
            return 'infeasible_plan'
        if float(row['adaptation_quality_score']) < 0.30:
            return 'weak_adaptation'
        if float(row['helpfulness_score']) < 0.30:
            return 'goal_drift'
        if float(row['semantic_similarity_score']) < 0.20:
            return 'low_reference_alignment'
        return 'pass'

    df['failure_type'] = df.apply(classify, axis=1)
    return df
