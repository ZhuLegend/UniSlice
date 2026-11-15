import itertools,json,tempfile,unittest,shutil,subprocess,sys,hashlib,os,tarfile,io
from pathlib import Path
from collections import Counter
from unittest.mock import patch
from run import ROOT,validate_fixed_batches,code_identity,git_metadata
from paper.common import prepare,verify_prepared,INPUTS
from paper.batching import choose,local_pairs,calibration_passes,archive_large_logs
from paper.batching_audit import fixed_sample_accounting,text_file
from paper import batching
from paper.analysis import batching_report
from paper import restore as runtime_restore

class PaperExperimentsTest(unittest.TestCase):
    def test_git_metadata_is_optional_for_source_archives(self):
        with tempfile.TemporaryDirectory()as tmp:
            self.assertEqual(git_metadata(Path(tmp)),{'revision':None,'worktree':None})
        with patch('run.subprocess.run',side_effect=FileNotFoundError):
            self.assertEqual(git_metadata(),{'revision':None,'worktree':None})

    def test_identity_uses_the_requested_binary_directory(self):
        with tempfile.TemporaryDirectory()as tmp:
            target=Path(tmp)/'custom';(target/'debug').mkdir(parents=True)
            names=['mock_virtual','mock_vgpu','scheduler_daemon','task_sender']
            for name in names:(target/'debug'/name).write_bytes(name.encode())
            with patch.dict(os.environ,{'CARGO_TARGET_DIR':str(target)}):
                identity=code_identity(Path(tmp))
            self.assertEqual(identity['binary_sha256'],{name:hashlib.sha256(name.encode()).hexdigest()for name in names})

    def test_fixed_targets_validate_model_coverage_types_and_range(self):
        cfg={'models':['a','b']}
        self.assertEqual(validate_fixed_batches({'a':1,'b':32},cfg),{'a':1,'b':32})
        for bad in [[],{'a':1},{'a':0,'b':2},{'a':1,'b':33},{'a':True,'b':2},{'a':1,'b':2.5}]:
            with self.assertRaises(ValueError):validate_fixed_batches(bad,cfg)

    def test_all_four_preparations_without_runtime_or_private_workspace(self):
        with tempfile.TemporaryDirectory()as tmp:
            for experiment in ['e0','e1','e2','batching']:
                out=Path(tmp)/experiment;result=prepare(experiment,out)
                self.assertFalse(result['started']);verify_prepared(out)
                with self.assertRaises(ValueError):prepare(experiment,out)
            e2=Path(tmp)/'e2';protocol=json.loads((e2/'protocol.json').read_text());self.assertEqual(sum(protocol['candidate_counts'].values()),416)
            self.assertEqual(sum(len(p.read_text().splitlines())for p in e2.glob('*/inputs/chunks/search-*.jsonl')),2080)
            e1=Path(tmp)/'e1';self.assertEqual(len(json.loads((e1/'commands.json').read_text())),5)
            replay=json.loads((e1/'inputs/replay.json').read_text());self.assertEqual((len(replay),sum(x[2]for x in replay)),(1483,24408))
            self.assertEqual(json.loads((e1/'protocol.json').read_text())['send_seconds'],30)
            e0=Path(tmp)/'e0';config=json.loads((e0/'inputs/configuration.json').read_text())
            self.assertEqual(Path(config['latency_csv']).parent,(e0/'inputs').resolve())
            commands=json.loads((e0/'commands.json').read_text())
            self.assertEqual([row['case']for row in commands],['1x','8x','20x'])
            self.assertEqual((config['create_delay_ms'],config['model_lifecycle']['load_ms']),(12545,836))
            for row in commands:
                self.assertIn('--paper-multiplier',row['command'])
                replay=json.loads((e0/'inputs'/f"replay-{row['case']}.json").read_text())
                self.assertEqual(len(replay),json.loads((e0/'protocol.json').read_text())['expected_requests'][row['case']])
            (e0/'inputs/batching-source.toml').write_text('changed')
            with self.assertRaises(ValueError):verify_prepared(e0)
            (e1/'inputs/replay.json').write_text('[]')
            with self.assertRaises(ValueError):verify_prepared(e1)

    def test_parallel_e2_requires_calibration_and_preserves_target_on_rejection(self):
        with tempfile.TemporaryDirectory()as tmp:
            out=Path(tmp)/'no-calibration'
            with self.assertRaises(ValueError):prepare('e2',out,2)
            self.assertFalse(out.exists())

    def test_portable_preparation_in_checkout_without_paper_workspace(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            for directory in ['scripts/simulation','scripts/exper_config','resource/simulation']:
                shutil.copytree(ROOT/directory,root/directory,ignore=shutil.ignore_patterns('__pycache__'))
            for filename in ['Cargo.toml','Cargo.lock'] :shutil.copy2(ROOT/filename,root/filename)
            (root/'logs').mkdir();shutil.copy2(ROOT/'logs/model_latency.csv',root/'logs/model_latency.csv')
            self.assertFalse((root/'paper-revision').exists())
            for experiment in ['e0','e1','e2','batching']:
                res=subprocess.run([sys.executable,str(root/'scripts/simulation/experiments.py'),'prepare',experiment,'--output',str(root/'outputs'/experiment)],cwd=root,capture_output=True,text=True)
                self.assertEqual(res.returncode,0,res.stderr)

    def test_bounded_search_and_two_sided_host_screen(self):
        self.assertEqual(len(list(itertools.product([1,2,4,8,16,32],repeat=2))),36)
        self.assertEqual(local_pairs([1,1]),[(1,1),(1,2),(2,1),(2,2)])
        for pair in [(1,32),(4,8),(16,32)]:self.assertIn(tuple(pair),local_pairs(pair));self.assertLessEqual(len(local_pairs(pair)),9)
        limits=json.loads((INPUTS/'batching/settings.json').read_text())['calibration_screen']
        self.assertTrue(calibration_passes(.9,.75,-2,limits));self.assertFalse(calibration_passes(.899,1,0,limits));self.assertFalse(calibration_passes(1,1,2.001,limits))

    def test_paper_runtime_restore_preserves_execution_sources(self):
        with tempfile.TemporaryDirectory()as tmp:
            for experiment in ['e0','e1','e2','batching']:
                out=Path(tmp)/experiment;result=runtime_restore.restore(experiment,out)
                self.assertFalse(result['started'])
                record=json.loads((out/'PAPER_RUNTIME.json').read_text())
                for name,digest in record['files'].items():
                    self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),digest)
                self.assertFalse((out/'paper-revision').exists())
                res=subprocess.run([sys.executable,str(out/'scripts/simulation/experiments.py'),'prepare',experiment,'--output',str(out/'runs')],cwd=out,capture_output=True,text=True)
                self.assertEqual(res.returncode,0,res.stderr)
                with self.assertRaises(ValueError):runtime_restore.restore(experiment,out)

    def test_paper_runtime_restore_rejects_unsafe_members_before_output_creation(self):
        for name,kind in [('../escape',tarfile.REGTYPE),('/escape',tarfile.REGTYPE),('link',tarfile.SYMTYPE)]:
            with tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);base=root/'resource/simulation/experiments/runtime';base.mkdir(parents=True)
                archive=base/'e0.tar.gz';payload=b'unsafe'
                with tarfile.open(archive,'w:gz')as bundle:
                    member=tarfile.TarInfo(name);member.type=kind;member.size=len(payload)if kind==tarfile.REGTYPE else 0
                    member.linkname='../escape';bundle.addfile(member,io.BytesIO(payload)if member.size else None)
                version={'archive':archive.name,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':{name:hashlib.sha256(payload).hexdigest()}}
                (base/'manifest.json').write_text(json.dumps({'experiments':{'e0':version}}))
                output=root/'output'
                with patch.object(runtime_restore,'ROOT',root),self.assertRaises(ValueError):runtime_restore.restore('e0',output)
                self.assertFalse(output.exists())

    def test_fixed_batch_transport_matches_current_and_frozen_runners(self):
        targets={'resnet50':4,'vgg19':8}
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'targets.json'
            flags,env=batching.fixed_batch_control(targets,path,'environment')
            self.assertEqual(flags,[]);self.assertEqual(json.loads(env['HSGS_EXPERIMENT_FIXED_BATCHES']),targets)
            self.assertFalse(path.exists())
            flags,env=batching.fixed_batch_control(targets,path,'cli')
            self.assertEqual(flags,['--experiment-fixed-batches',str(path)]);self.assertEqual(env,{})
            self.assertEqual(json.loads(path.read_text()),targets)
            with self.assertRaises(ValueError):batching.fixed_batch_control(targets,path,'unknown')

    def test_selection_keeps_incomplete_candidates_and_uses_common_requests(self):
        def row(pair,requests):return {'fixed_targets':dict(zip(['resnet50','vgg19'],pair)),'audit':{'requests':requests,'metrics':{'all':{'slo400_percent':50,'completion_percent':50,'censored':1,'all_offered_capped_e2e_mean_ms':2550,'successful_server_mean_ms':100}}}}
        a=[{'request_id':'a','observed_complete':True,'server_jct_ms':100},{'request_id':'b','observed_complete':False,'server_jct_ms':None}]
        b=[{'request_id':'a','observed_complete':True,'server_jct_ms':120},{'request_id':'b','observed_complete':True,'server_jct_ms':1}]
        result=choose([row((1,1),a),row((2,2),b)])
        self.assertEqual(result['selected']['pair'],[1,1]);self.assertEqual(result['common_request_ids'],['a'])
        v=fixed_sample_accounting(Counter(m=19),Counter(m=8),{'m':4})['m'];self.assertEqual(v['excess_over_modulus_samples'],8)

    def test_archive_is_lossless(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(batching,'S',Path(tmp)):
            run=Path(tmp)/'run';run.mkdir();source=run/'scheduler.log';payload=('原始记录\n'*30000).encode();source.write_bytes(payload)
            archive_large_logs(run,lambda:None)
            self.assertFalse(source.exists());self.assertEqual(text_file(source).encode(),payload)
            record=json.loads((run.parent/'log-archives.json').read_text())[0];self.assertEqual(record['decompressed_sha256'],hashlib.sha256(payload).hexdigest())

    def test_validation_analysis_rejects_extra_seeds_and_keeps_switching_windows(self):
        protocol={'cases':{'switching':{'seconds':90}},'validation_seeds':[11]}
        requests=[{'request_id':str(i),'phase':phase,'observed_complete':True,'server_jct_ms':100+i,'end_to_end_ms':110+i}for i,phase in enumerate(['low_before','high','low_after'])]
        rows=[{'case':'switching','runtime_seed':11,'mode':mode,'audit':{'requests':requests}}for mode in ['dynamic','fixed']]
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);path=root/'validation-results.json';path.write_text(json.dumps(rows))
            result=batching_report(root,protocol);self.assertEqual(result[0]['common_requests'],3)
            self.assertEqual(len((root/'paired-phase-metrics.csv').read_text().splitlines()),4)
            path.write_text(json.dumps(rows+[{**r,'runtime_seed':12}for r in rows]))
            with self.assertRaises(ValueError):batching_report(root,protocol)

if __name__=='__main__':unittest.main()
