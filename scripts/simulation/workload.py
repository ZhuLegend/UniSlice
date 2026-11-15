"""Seeded multi-sample input using the original sender's sample-rate semantics."""
import math
import random

# ponytail: keep replay construction bounded; stream input if traces exceed this ceiling.
MAX_REPLAY_EVENTS = 1_000_000


def phase_events(models, rates, index, seconds, seed, batch_range, remaining):
    low, high = batch_range
    if not 1 <= low <= high <= 32:
        raise ValueError('input batch range must be within [1, 32]')
    events = []
    for model in sorted(models):
        rng = random.Random(f'{seed}:{index}:{model}')
        t = 0.0
        while True:
            batch = rng.randint(low, high)
            # Original sender: draw B first, then an exponential interval of mean B/S.
            t += rng.expovariate(rates[model] / batch)
            if t >= seconds:
                break
            if len(events) >= remaining:
                raise ValueError('replay exceeds the one-million-request memory ceiling')
            events.append([index * seconds + t, model, batch])
    return events


def long_cycle_workload(models, multiplier, phase_seconds=360, seed=20260923, batch_range=(1, 32)):
    """Six steady phases with five changes, including two different hot model sets."""
    if not math.isfinite(multiplier) or multiplier <= 0 or not math.isfinite(phase_seconds) or phase_seconds <= 0:
        raise ValueError('multiplier and phase duration must be positive')
    base = {model: 12.5 * multiplier for model in models}
    definitions = [
        ('balanced', '均衡', set()),
        ('vgg_double', 'VGG 翻倍', {'vgg16', 'vgg19'}),
        ('balanced_restore_1', '恢复均衡①', set()),
        ('resnet_double', 'ResNet 翻倍', {'resnet18', 'resnet50'}),
        ('vgg_double_2', '热点转到 VGG', {'vgg16', 'vgg19'}),
        ('balanced_restore_2', '恢复均衡②', set()),
    ]
    phases, events = [], []
    for index, (name, label, hot) in enumerate(definitions):
        rates = {model: rate * (2 if model in hot else 1) for model, rate in base.items()}
        phases.append({'start_seconds': index * phase_seconds, 'name': name, 'label': label, 'rates': rates})
        events.extend(phase_events(models, rates, index, phase_seconds, seed, batch_range, MAX_REPLAY_EVENTS-len(events)))
    events.sort(key=lambda row: (row[0], row[1]))
    return {'name': 'long-cycle-multi-slice', 'multiplier': multiplier, 'seed': seed,
            'duration_seconds': len(phases) * phase_seconds, 'initial_rates': base,
            'slo_ms': [250, 400], 'batch_size_range': list(batch_range), 'rate_unit': 'samples/s', 'phases': phases,
            'provenance': 'Extended stability experiment, not a paper reproduction: 12.5 samples/s/model base from original sender semantics; '
                          'uniform input batches and exponential intervals conditional on batch. Six phases with VGG and ResNet hotspots; '
                          '250/400ms SLO. Slice profiles are chosen by the existing scheduler.'}, events


def paper_workload(models, multiplier, phase_seconds=60, seed=20260923, batch_range=(1, 32)):
    if not math.isfinite(multiplier) or multiplier <= 0 or not math.isfinite(phase_seconds) or phase_seconds <= 0:
        raise ValueError('multiplier and phase duration must be positive')
    # The paper specifies a range, not an exact per-model trace. Use its midpoint.
    base = {model: 12.5 * multiplier for model in models}
    phases = []
    events = []
    for index, name in enumerate(['balanced', 'vgg_double', 'balanced_restore']):
        rates = {m: rate * (2 if index == 1 and m in {'vgg16', 'vgg19'} else 1)
                 for m, rate in base.items()}
        phases.append({'start_seconds': index * phase_seconds, 'name': name, 'rates': rates})
        events.extend(phase_events(models, rates, index, phase_seconds, seed, batch_range, MAX_REPLAY_EVENTS-len(events)))
    events.sort(key=lambda row: (row[0], row[1]))
    return {'name': 'paper-dynamic', 'multiplier': multiplier, 'seed': seed,
            'duration_seconds': 3 * phase_seconds, 'initial_rates': base,
            'slo_ms': [250, 400], 'batch_size_range': list(batch_range), 'rate_unit': 'samples/s', 'phases': phases,
            'provenance': 'UniSlice-02-04 Table II: 250/400 ms; input follows original sender samples/s and uniform batch 1–32, '
                          'which differs from the paper wording requests/s. Midpoint 12.5, seeded conditional exponential arrivals, three phases. '
                          'VGG doubling follows the imbalance ratio in IX-B; 4x is an additional load point.'}, events
