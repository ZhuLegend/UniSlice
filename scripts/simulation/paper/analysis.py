"""Original-request metrics; every generated request remains in the SLO denominator."""
import csv,json,statistics
from collections import defaultdict,Counter
from pathlib import Path
from run import percentile
from long_analysis import log_events
from .common import read,save

def write_csv(path,rows):
    if rows:
        with path.open('w',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def feedback(run):
    cfg=read(run/'simulation.json');curves=defaultdict(dict)
    with Path(cfg['latency_csv']).open()as f:
        for r in csv.DictReader(f):curves[r['model_name'].lower(),r['split_param'].lower()][int(r['batch_size'])]=float(r['avg_latency(s)'])
    count=Counter();data=[];start=read(run/'measurement/final_summary.json')['planned_origin_ms']
    for e in log_events(run/'scheduler/lifecycle_events.jsonl'):
        if e.get('event')!='online_prediction_feedback':continue
        key=e['pool_id'],e['model'],e['profile'];count[key]+=1;curve=curves[e['model'],cfg.get('profile_aliases',{}).get(e['profile'],e['profile'])];b=e['batch']
        if b in curve:truth=curve[b]
        else:
            lo=max(k for k in curve if k<b);hi=min(k for k in curve if k>b);truth=curve[lo]+(b-lo)*(curve[hi]-curve[lo])/(hi-lo)
        before,observed=e['prediction_before_seconds'],e['observed_processing_seconds']
        data.append({'pool':key[0],'model':key[1],'profile':key[2],'batch':b,'observation_index':count[key],'offset_s':(e['timestamp_ms']-start)/1000,'truth_error':before/truth-1,'processing_error':before/observed-1})
    if not data:raise ValueError('no online feedback records')
    write_csv(run/'online-prediction-feedback.csv',data)
    result={'observations':len(data),'executed_configurations':len(count)}
    for field in ['truth_error','processing_error']:
        for name,rows in [('all',data),('after_10',[r for r in data if r['observation_index']>10])]:
            vals=[100*abs(r[field])for r in rows];result[field+'_'+name]={'n':len(vals),'mean_absolute_percent':statistics.mean(vals)if vals else None,'p99_absolute_percent':percentile(vals,.99)if vals else None,'max_absolute_percent':max(vals,default=None)}
    save(run/'prediction-error-analysis.json',result);return result

def batching_report(root,protocol):
    groups=defaultdict(dict)
    for row in read(root/'validation-results.json'):
        key=row['case'],row['runtime_seed'];mode=row['mode']
        if mode in groups[key]:raise ValueError('duplicate validation arm')
        groups[key][mode]=row
    expected={(case,seed)for case in protocol['cases']for seed in protocol['validation_seeds']}
    if set(groups)!=expected:raise ValueError('unexpected or missing validation case/seed')
    rows=[];details=[];phases=[]
    for case in protocol['cases']:
        measured=[]
        for seed in protocol['validation_seeds']:
            pair=groups[case,seed]
            if set(pair)!={'dynamic','fixed'}:raise ValueError('incomplete matched validation pair')
            requests={m:{r['request_id']:r for r in x['audit']['requests']}for m,x in pair.items()}
            if any(len(requests[m])!=len(pair[m]['audit']['requests'])for m in pair):raise ValueError('duplicate request identity')
            if set(requests['dynamic'])!=set(requests['fixed']):raise ValueError('offered request identities differ')
            for window in ['all']+(['low_before','high','low_after']if protocol['cases'][case]['seconds']==90 else []):
                scoped={m:{k:v for k,v in rr.items()if window=='all'or v['phase']==window}for m,rr in requests.items()}
                if set(scoped['dynamic'])!=set(scoped['fixed']):raise ValueError('request phases differ')
                common=set.intersection(*[{k for k,v in rr.items()if v['observed_complete']}for rr in scoped.values()])
                if not common:raise ValueError('no jointly completed requests')
                item={'case':case,'seed':seed,'window':window,'common_requests':len(common)}
                for mode,rr in scoped.items():
                    vals=[rr[k]['server_jct_ms']for k in common]
                    item.update({mode+'_mean_ms':statistics.mean(vals),mode+'_p99_ms':percentile(vals,.99),mode+'_slo400_percent':100*sum(v['observed_complete']and v['end_to_end_ms']<=400 for v in rr.values())/len(rr),mode+'_unfinished':sum(not v['observed_complete']for v in rr.values())})
                if window=='all':measured.append(item);details.append(item)
                else:phases.append(item)
        row={'case':case,'validation_pairs':len(measured)}
        for key in measured[0]:
            if key not in ['case','seed','window']:row[key]=statistics.mean(x[key]for x in measured)
        row['mean_change_percent']=100*(row['dynamic_mean_ms']/row['fixed_mean_ms']-1);rows.append(row)
    save(root/'paired-validation-metrics.json',details)
    write_csv(root/'paired-phase-metrics.csv',phases)
    return rows

def analyze(experiment,root):
    protocol=read(root/'protocol.json')
    if protocol['experiment']!=experiment:raise ValueError('wrong experiment')
    if experiment=='batching':rows=batching_report(root,protocol)
    elif experiment=='e2':
        rows=[]
        for case in read(root/'suite-results.json')['scenarios']:
            v=case['validation'];rows.append({'case':case['name'],'candidates':case['search']['candidate_count'],'greedy_mean_ms':v['greedy']['mean_jct_ms'],'optimal_mean_ms':v['reference']['mean_jct_ms'],'gap_percent':v['jct_gap_percent'],'greedy_p99_ms':v['greedy']['mean_per_run_p99_ms'],'optimal_p99_ms':v['reference']['mean_per_run_p99_ms'],'greedy_slo400_percent':v['greedy']['mean_slo_400_percent'],'optimal_slo400_percent':v['reference']['mean_slo_400_percent'],'same_allocation':v['same_allocation']})
    else:
        rows=[]
        for job in read(root/'commands.json'):
            run=root/job['case'];summary=read(run/'experiment-metrics.json');clients=[json.loads(s)for s in(run/'measurement/client-terminal.jsonl').read_text().splitlines()if s.strip()];good=[c for c in clients if c['success']]
            row={'case':job['case'],'offered_requests':len(clients),'completed':len(good),'failed':len(clients)-len(good),'mean_jct_ms':summary['mean_jct_ms'],'p99_jct_ms':summary['p99_jct_ms'],'slo400_percent':100*sum(c['end_to_end_ms']<=400 for c in good)/len(clients)}
            if experiment=='e1':feedback(run)
            rows.append(row)
    result={'experiment':experiment,'cases':rows,'runtime_scope':'Metrics from the supplied run records; runtime provenance is recorded separately in build-identity.json and each run manifest.'}
    save(root/'report.json',result);write_csv(root/'summary.csv',rows)
    lines=[f'# {experiment} results','',result['runtime_scope'],'']
    if rows:
        keys=list(rows[0]);lines+=['| '+' | '.join(keys)+' |','|'+'|'.join('---'for _ in keys)+'|']+['| '+' | '.join(str(r[k])for k in keys)+' |'for r in rows]
    (root/'report.md').write_text('\n'.join(lines)+'\n');return result
