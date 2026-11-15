import json,os,shutil,subprocess,sys,time
from pathlib import Path
from run import ROOT,code_identity,dump
from .archive import archive,sha
from .allocation_audit import audit

def read(p):return json.loads(p.read_text())
def save(p,v):
    tmp=Path(str(p)+".tmp");dump(tmp,v);os.replace(tmp,p)

class Suite:
    def __init__(self,root,resume):
        self.root=root
        self.identity=read(root/'build-identity.json')['code_identity']
        self.state_path=root/'suite-status.json'
        self.manifest_path=root/'suite-manifest.json'
        scripts=list(Path(__file__).parent.glob('*.py'))
        inputs=[p for p in root.glob('*/inputs/**/*') if p.is_file() and p.suffix in {'.json','.jsonl','.toml'}]
        files={str(path.resolve()):sha(path) for path in scripts+inputs}
        if self.manifest_path.exists():
            assert resume,'existing suite requires explicit --resume'
            self.manifest=read(self.manifest_path)
            assert self.manifest['code_identity']==self.identity
            self.state=read(self.state_path)
        else:
            self.manifest={'code_identity':self.identity,'frozen_files':files,'protocol':read(root/'protocol.json'),
                'storage':'Lossless archive after completed queue audit; verify regular-file SHA-256 and sizes before releasing raw trial copies.',
                'restart':'Completed chunks can be reused with --resume after identity checks. An incomplete chunk stops for review; no silent retry or individual-job recovery.',
                'concurrency_gate':'Serial by default. Parallel execution requires an explicitly reviewed gate_passed calibration report copied into this output directory.'}
            save(self.manifest_path,self.manifest)
            self.state={'status':'running','started_at_ms':round(time.time()*1000),'stages':{},'active':None,'error':None}
        self.process=None
        self.verify()
        save(root/'supervisor.json',{'pid':os.getpid(),'command':sys.argv,'started_at_ms':round(time.time()*1000)})


    def verify(self):
        assert code_identity()==self.identity,'runtime source or binaries changed'
        for name,digest in self.manifest['frozen_files'].items():
            assert sha(Path(name))==digest,f'frozen experiment input/tool changed: {name}'


    def checkpoint(self):
        self.state['updated_at_ms']=round(time.time()*1000)
        save(self.state_path,self.state)


    def queue(self,name,label,jobs,concurrency=1,pack=False):
        key=f'{name}/{label}';folder=self.root/name/label
        stage=self.state['stages'].get(key)
        if stage and stage['status']=='complete':
            summary=read(folder/'summary.json');assert summary['finished'] and not summary['error']
            certificate=read(folder/'coverage-audit.json')
            assert all(sha(folder/file)==digest for file,digest in certificate['input_sha256'].items())
            if pack:
                archive(folder)
            return certificate
        self.verify()
        assert shutil.disk_usage(self.root).free>=8*1024**3,'less than 8 GiB free; stop before launching next queue'
        self.state['active']={'scenario':name,'stage':label,'completed_trials':0,'concurrency':concurrency}
        self.state['stages'][key]={'status':'running','jobs':str(jobs),'output':str(folder)}
        self.checkpoint()
        if folder.exists():
            summary=read(folder/'summary.json')
            assert summary['finished'] and not summary['error'],'incomplete previous queue preserved; review before retry'
        else:
            inputs=self.root/name/'inputs'
            command=[sys.executable,str(ROOT/'scripts/simulation/paper/queue.py'),
                '--jobs',str(jobs),'--config',str(inputs/'configuration.json'),
                '--replay',str(inputs/'replay.json'),'--model-rates',str(inputs/'model-rates.json'),
                '--output',str(folder),'--concurrency',str(concurrency),'--rate','75',
                '--seconds','30','--noise-cv','.1','--seed','20260923','--base-port','24300']
            save(self.root/name/f'{label}-command.json',command)
            print('starting',key,'concurrency',concurrency,flush=True)
            with (self.root/name/f'{label}-console.log').open('w') as log:
                self.process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                self.state['active']['queue_pid']=self.process.pid;self.checkpoint()
                last=-1
                while self.process.poll() is None:
                    try:
                        summary=read(folder/'summary.json')
                        completed=summary['completed_trials']
                        self.state['active'].update(completed_trials=completed,launched_trials=summary['launched_trials'],wall_seconds=summary['wall_seconds'])
                        self.checkpoint()
                        if int(time.monotonic()/60)!=last:
                            print(key,'completed',completed,'launched',summary['launched_trials'],flush=True)
                            last=int(time.monotonic()/60)
                    except (FileNotFoundError,json.JSONDecodeError):pass
                    time.sleep(2)
                result=self.process.returncode;self.process=None
            assert result==0,f'queue failed: {key}; inspect its console and preserved trial directories'
        certificate,_=audit(folder)
        assert read(folder/'manifest.json')['code_identity']==self.identity
        self.verify()
        if pack:
            saved=archive(folder)
            certificate['archive_bytes']=saved['archive_bytes']
            certificate['raw_trial_bytes']=saved['raw_bytes']
            save(folder/'coverage-audit.json',certificate)
        self.state['stages'][key]={'status':'complete','completed_trials':certificate['completed'],
            'application_failure_trials':certificate['complete_application_failure_trials'],'wall_seconds':certificate['wall_seconds'],
            'output':str(folder),'archived':pack}
        self.state['active']=None;self.checkpoint()
        print('completed',key,certificate['completed'],round(certificate['wall_seconds'],2),flush=True)
        return certificate


    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=370)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
