"""Verify anonymous relay streaming and cached replay against an audited run."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--origin', default='https://mukundshankar-dev.github.io')
    parser.add_argument('--examples', type=Path, default=Path('demo/web/examples.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.examples.read_text())[0]
    headers = {'Origin': args.origin}
    started = perf_counter()
    lengths = []
    first_seconds = None
    with httpx.stream('POST', args.url, json=expected['request'], headers=headers, timeout=130) as response:
        response.raise_for_status()
        assert response.headers['access-control-allow-origin'] == args.origin
        for line in response.iter_lines():
            if not line:
                continue
            frame = json.loads(line)
            if 'error' in frame:
                raise RuntimeError(frame['error'])
            if 'trace' in frame:
                if first_seconds is None:
                    first_seconds = perf_counter() - started
                lengths.append(len(frame['trace']))
                final = frame
    elapsed = perf_counter() - started
    assert final['status'] == 'complete'
    assert final['summary'] == expected['summary']
    assert [r['prediction'] for r in final['trace']] == [r['prediction'] for r in expected['trace']]
    assert [r['stop_probability'] for r in final['trace']] == [r['stop_probability'] for r in expected['trace']]
    replay = httpx.post(args.url, json=expected['request'], headers=headers, timeout=10)
    replay.raise_for_status()
    assert replay.json()['served_from_cache']
    assert replay.json()['trace'] == final['trace']
    assert httpx.post(args.url, json={}, headers=headers).status_code == 400
    assert httpx.post(args.url, json=expected['request']).status_code == 403
    result = {'url': args.url, 'authenticated_browser': False, 'frames': lengths,
              'incremental': len(set(lengths)) > 1, 'first_frame_seconds': first_seconds,
              'wall_seconds': elapsed, 'cached_replay': True,
              'audited_predictions_and_stop_probabilities_match': True,
              'invalid_request_rejected': True, 'missing_origin_rejected': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
