"""Summarize committed training-evaluation histories; does not run a policy."""
import argparse
import json
from pathlib import Path


def summarize(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    history = data.get('eval_history', [])
    if not history:
        raise ValueError(f'{path}: missing evaluation history')
    if any(not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], (int, float))
           or not isinstance(row[1], (int, float)) or not 0 <= row[1] <= 1 for row in history):
        raise ValueError(f'{path}: expected [iteration, accuracy in 0..1] rows')
    best = max(history, key=lambda row: row[1])
    return {'run': data.get('exp_name', Path(path).parent.name),
            'best_iteration': best[0], 'best_sort_accuracy': best[1],
            'last_iteration': history[-1][0], 'last_sort_accuracy': history[-1][1],
            'evaluations': len(history), 'source': str(path),
            'measurement': 'training-time evaluation; not a held-out competition score'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', nargs='+', type=Path)
    args = p.parse_args()
    try:
        print(json.dumps([summarize(path) for path in args.results], indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
