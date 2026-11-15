import json
from itertools import product
PROFILES = [("1g.10gb",1,10)]
SLICE = "1g.10gb"

def plans(models, template='full', ce_budget=7, memory_budget=80, profiles=PROFILES, require_coverage=True):
    choices = [(model, name, ce, memory) for model in models for name, ce, memory in profiles]
    placements = []
    def visit(index, ce_left, memory_left):
        if index == len(choices):
            if not require_coverage or {row['model'] for row in placements} == set(models):
                yield {'schema_version':1,'placements':[dict(row) for row in placements]}
            return
        model, name, ce, memory = choices[index]
        for count in range(min(ce_left//ce,memory_left//memory)+1):
            if count:
                placements.append({'template_name':template,'model':model,'slice':name,'count':count})
            yield from visit(index+1,ce_left-count*ce,memory_left-count*memory)
            if count:
                placements.pop()
    yield from visit(0,ce_budget,memory_budget)


def write_jobs(path, rows):
    with path.open('x') as output:
        for row in rows:
            output.write(json.dumps(row,sort_keys=True)+'\n')

def restricted_plans(models, nodes):
    domains = [list(plans(models, node['template_name'], node['resource']['ce'],
                          node['resource']['memory'], [(SLICE, 1, 10)], False))
               for node in nodes]
    for parents in product(*domains):
        placements = [p for parent in parents for p in parent['placements']]
        if {p['model'] for p in placements} == set(models):
            yield {'schema_version': 1, 'placements': placements}

