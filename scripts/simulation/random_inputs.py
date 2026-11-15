"""Resolve explicit final random seeds; schema 1 uses SHA-256 seed labels."""
import hashlib
import uuid


def final_seed(root_seed, label):
    return int.from_bytes(hashlib.sha256(f'HSGS-random-v1\0{root_seed}\0{label}'.encode()).digest()[:8], 'big')


def resolve_random_inputs(config, root_seed, worker_slots=64, model_loads=16):
    if type(root_seed) is not int or not 0 <= root_seed < 2**64:
        raise ValueError('random seed must fit u64')
    if not 1 <= worker_slots <= 4096 or not 1 <= model_loads <= 4096:
        raise ValueError('explicit noise budgets must be in [1,4096]')
    models = sorted(config['models'])
    seeds, worker_noise = {}, {}
    for model in models:
        key = f'routing/{model}'
        seeds[key] = final_seed(root_seed, key)
    for index, template in enumerate(sorted({node['template_name'] for node in config['nodes']})):
        pool = str(uuid.UUID(int=0x48534753000000000000000000000000 | index))
        for model in models:
            key = f'extraction/{pool}/{model}'
            seeds[key] = final_seed(root_seed, key)
    for node in sorted(config['nodes'], key=lambda node: node['name']):
        for gpu in range(node['gpus']):
            for profile in sorted(node['share_targets']):
                for slot in range(worker_slots):
                    key = f"{node['name']}/{gpu}/{profile}/{slot}"
                    worker_noise[key] = {model: [final_seed(root_seed, f'noise/{key}/{model}/{generation}')
                                               for generation in range(model_loads)] for model in models}
                    for model in models:
                        rounding_key = f'rounding/{key}/{model}'
                        seeds[rounding_key] = final_seed(root_seed, rounding_key)
    return {'schema_version': 1, 'root_seed': root_seed, 'seeds': seeds, 'worker_noise': worker_noise}
