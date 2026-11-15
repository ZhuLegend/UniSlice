"""Prepare portable inputs; execution always records a new runtime identity."""
from pathlib import Path
import json,os,shutil,signal,subprocess,sys
from run import ROOT,code_identity,source_fingerprint,scheduler_config,toml_inline
from .domain import restricted_plans,write_jobs
from .allocation import search_jobs,SEARCH_SEEDS,VALIDATION_SEEDS
from .archive import sha
from workload import long_cycle_workload

INPUTS=ROOT/'resource/simulation/experiments'
def read(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def copy_inputs(source,dest):
    dest.mkdir(parents=True)
    for p in source.iterdir():
        if p.is_file():shutil.copy2(p,dest/p.name)
    config=read(dest/'configuration.json')
    for key in ['latency_csv','batching_config']:config[key]=str((dest/config[key]).resolve())
    save(dest/'configuration.json',config)
    return config

def prepare(experiment,root,concurrency=1,calibration=None):
    if root.exists():raise ValueError('output already exists; preserve it and choose a fresh directory')
    if concurrency<1:raise ValueError('concurrency must be positive')
    settings=read(INPUTS/experiment/'settings.json')
    if experiment=='e2'and concurrency>1:
        if not calibration or not read(calibration).get('gate_passed'):
            raise ValueError('parallel E2 needs a reviewed calibration report with gate_passed=true')
    root.mkdir(parents=True)
    protocol={**settings,'experiment':experiment,'source_sha256':source_fingerprint(),
        'concurrency':concurrency,'scope':'New measurements on current source; historical input traces do not relabel this runtime as a historical measurement.'}
    if experiment=='e0':
        cfg=copy_inputs(INPUTS/'e0/current',root/'inputs')
        protocol['retained_replay_sha256']={}
        for multiplier in settings['paper_multipliers']:
            name=f'{multiplier}x';path=root/'inputs'/f'replay-{name}.json'
            _,events=long_cycle_workload(cfg['models'],multiplier,settings['phase_seconds'],settings['seed'],(settings['input_batch_min'],settings['input_batch_max']))
            if events!=read(path):raise ValueError('generated E0 arrivals differ from retained paper replay: '+name)
            protocol['retained_replay_sha256'][name]=sha(path)
    elif experiment=='e2':
        total=0
        for name in settings['scenarios']:
            inp=root/name/'inputs';cfg=copy_inputs(INPUTS/experiment/name,inp);(inp/'chunks').mkdir()
            candidates=[{'id':f'c{i:04d}','allocation':p}for i,p in enumerate(restricted_plans(sorted(cfg['models']),cfg['nodes']),1)]
            assert len(candidates)==settings['candidate_counts'][name];total+=len(candidates)
            write_jobs(inp/'nominal-jobs.jsonl',candidates)
            write_jobs(inp/'native-check-jobs.jsonl',[{'id':f'native-r{i}','policy':'greedy','runtime_seed':s}for i,s in enumerate(SEARCH_SEEDS,1)])
            for i,seed in enumerate(SEARCH_SEEDS,1):write_jobs(inp/f'chunks/search-r{i}.jsonl',search_jobs(candidates,seed,i))
            save(inp/'coverage.json',{'nominal_candidates':len(candidates),'enumeration':'every feasible integer placement, including unused capacity; no objective pruning'})
        assert total==416
        protocol.update(concurrency_gate='Serial by default; parallel limit requires an explicit calibration report.',serial_repeats=10)
        save(root/'provenance.json',{'input_files':{str(p.relative_to(root)):sha(p)for p in root.glob('*/inputs/*')if p.is_file()}})
        if calibration:shutil.copy2(calibration,root/'concurrency-calibration.json')
    elif experiment=='batching':
        runtime=ROOT/'PAPER_RUNTIME.json'
        protocol['fixed_batch_control']='environment'if runtime.exists()and read(runtime).get('experiment')=='batching'else'cli'
        protocol['tuning_concurrency']=concurrency
        protocol['target']=str(Path(os.environ.get('CARGO_TARGET_DIR',ROOT/'target')).resolve())
        for name in settings['cases']:
            inp=root/name/'inputs';cfg=copy_inputs(INPUTS/experiment/name,inp)
            for mode in ['dynamic','fixed']:
                save(inp/f'configuration-{mode}.json',{**cfg,'batching_config':str(inp/f'batching-{mode}.toml')})
    elif experiment=='e1':
        copy_inputs(INPUTS/'e1/current',root/'inputs')
        (root/'predictions').mkdir()
        for name in settings['cases']:
            overrides=read(INPUTS/'e1/configs'/f'{name}.json');snapshots={}
            for template,source in overrides.items():
                p=INPUTS/'e1/configs'/source;dest=root/'predictions'/f'{name}-{template}.csv';shutil.copy2(p,dest);snapshots[template]=str(dest)
            save(root/'predictions'/f'{name}.json',snapshots)
    else:raise ValueError('unknown experiment')
    save(root/'protocol.json',protocol)
    commands=[]
    if experiment in ['e0','e1']:
        cases=settings['cases']if experiment=='e1'else[f'{value}x'for value in settings['paper_multipliers']]
        for name in cases:
            cmd=[sys.executable,str(ROOT/'scripts/simulation/run.py'),'--skip-build','--output',str(root/name),'--warmup-seconds','0','--client-terminal-log','--seed','20260923','--noise-seed','20260923','--noise-cv','.1','--schedule-interval','15']
            if experiment=='e1':
                cmd+=['--config',str(root/'inputs/configuration.json'),'--replay',str(root/'inputs/replay.json'),'--model-rates',str(root/'inputs/model-rates.json'),'--seconds','30','--experiment-freeze-greedy','--prediction-csvs',str(root/'predictions'/f'{name}.json'),'--routing-snapshot-seconds','1']
            else:cmd+=['--config',str(root/'inputs/configuration.json'),'--paper-multiplier',name[:-1],'--long-cycle','--phase-seconds','360','--input-batch-min','1','--input-batch-max','32','--random-seed','20260923','--random-model-loads','32','--routing-snapshot-seconds','1']
            commands.append({'case':name,'command':cmd})
    save(root/'commands.json',commands)
    save(root/'prepared-inputs.json',{str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()})
    return {'experiment':experiment,'prepared':str(root),'cases':len(settings.get('scenarios',settings.get('cases',settings.get('paper_multipliers',[])))),'started':False}

def verify_prepared(root):
    for name,digest in read(root/'prepared-inputs.json').items():
        if sha(root/name)!=digest:raise ValueError(f'prepared input changed: {name}')
    protocol=read(root/'protocol.json')
    if protocol['source_sha256']!=source_fingerprint():raise ValueError('source changed after preparation; prepare a new directory')
    return protocol

def execute(experiment,root,resume=False):
    protocol=verify_prepared(root)
    if protocol['experiment']!=experiment:raise ValueError('experiment and prepared directory differ')
    identity=code_identity()
    identity_path=root/'build-identity.json'
    if identity_path.exists():
        if read(identity_path)['code_identity']!=identity:raise ValueError('prepared runtime binaries changed')
    else:save(identity_path,{'code_identity':identity})
    if experiment=='e2':
        from .allocation import run
        run(root,resume,protocol['concurrency'])
    elif experiment=='batching':
        if resume:raise ValueError('batching keeps interrupted runs; inspect and prepare a fresh directory')
        from .batching import Pipeline
        save(root/'experiment-identity.json',{'runtime':identity,'files':{str(root/p):h for p,h in read(root/'prepared-inputs.json').items()}})
        protocol.update(runtime_identity=identity);save(root/'protocol.json',protocol)
        # Protocol now includes built binary identities; freeze the updated bytes.
        cert=read(root/'experiment-identity.json');cert['files'][str(root/'protocol.json')]=sha(root/'protocol.json');save(root/'experiment-identity.json',cert)
        prepared=read(root/'prepared-inputs.json');prepared['protocol.json']=sha(root/'protocol.json');save(root/'prepared-inputs.json',prepared)
        Pipeline(root).run()
    else:
        from terminal import audit
        state=root/'suite-status.json'
        if state.exists():raise ValueError('suite already started; preserve it and use a fresh directory')
        save(state,{'status':'running','completed':[]})
        done=[]
        try:
            for job in read(root/'commands.json'):
                log=root/(job['case']+'-runner.log')
                with log.open('x')as handle:
                    proc=subprocess.Popen(job['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'},start_new_session=True,close_fds=True)
                    def interrupted(signum,_frame):
                        raise KeyboardInterrupt(f'experiment interrupted by signal {signum}')
                    previous={sig:signal.signal(sig,interrupted)for sig in [signal.SIGINT,signal.SIGTERM]}
                    try:
                        code=proc.wait()
                    except BaseException:
                        if proc.poll()is None:
                            proc.terminate();proc.wait(timeout=370)
                        raise
                    finally:
                        for sig,handler in previous.items():signal.signal(sig,handler)
                _,summary,certificate=audit(root/job['case'],code,log)
                if experiment=='e0'and sha(root/job['case']/'replay.json')!=protocol['retained_replay_sha256'][job['case']]:
                    raise ValueError('executed E0 replay differs from retained paper input')
                save(root/job['case']/'experiment-metrics.json',summary)
                save(root/job['case']/'terminal-audit.json',certificate);done.append(job['case']);save(state,{'status':'running','completed':done})
            save(state,{'status':'complete','completed':done})
        except BaseException as error:
            save(state,{'status':'stopped','completed':done,'error':str(error)});raise
