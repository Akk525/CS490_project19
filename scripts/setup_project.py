from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        'data/interim',
        'data/processed',
        'data/retrieval_library',
        'data/splits',
        'outputs/generations',
        'outputs/retrieval',
        'outputs/evaluations',
        'outputs/summaries',
        'outputs/analysis',
        'outputs/logs',
        'outputs/manifests',
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for keep in [
        'data/interim/.gitkeep',
        'data/processed/.gitkeep',
        'data/retrieval_library/.gitkeep',
        'data/splits/.gitkeep',
        'outputs/.gitkeep',
    ]:
        p = root / keep
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)
    print('Project directories initialized.')


if __name__ == '__main__':
    main()
