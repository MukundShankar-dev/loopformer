import json

from scripts.training.tracking import Tracker, scalar_metrics, tracking_run


def test_scalar_mapping_and_same_optimizer_step():
    class Run:
        def __init__(self):
            self.events = []
        def log(self, event):
            self.events.append(event)
    run = Run()
    tracker = Tracker(run)
    tracker.log({'event': 'train', 'step': 4, 'train': {'loss': 1, 'per_loop': {'2': {'accuracy': .5}}}})
    tracker.log({'event': 'validation', 'step': 4, 'validation': {'selection_loss': .3}})
    assert [x['optimizer_step'] for x in run.events] == [4, 4]
    assert run.events[0]['train/per_loop/2/accuracy'] == .5
    assert run.events[1]['validation/selection_loss'] == .3
    assert scalar_metrics({'undefined': None, 'text': 'x', 'list': [1]}) == {}


def test_disabled_has_no_files(tmp_path):
    with tracking_run(tmp_path, {}, {}, mode='disabled', project='loopformer') as tracker:
        tracker.log({'event': 'train', 'step': 0})
    assert list(tmp_path.iterdir()) == []


def test_offline_sdk_lifecycle(tmp_path):
    with tracking_run(tmp_path, {'resume': 'parent/step-5'}, {'seed': 17},
                      mode='offline', project='loopformer') as tracker:
        tracker.log({'event': 'train', 'step': 6, 'train': {'loss': .2}})
        tracker.log({'event': 'validation', 'step': 6, 'validation': {'selection_loss': .3}})
        tracker.summary({'step': 6, 'training_seconds': 1})
        assert tracker.run.summary['final/step'] == 6
    saved = json.loads((tmp_path / 'wandb_run.json').read_text())
    assert saved['project'] == 'loopformer'
    assert saved['mode'] == 'offline'
    assert list((tmp_path / 'wandb').glob('offline-run-*/*.wandb'))
