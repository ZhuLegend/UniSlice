#!/usr/bin/env python3
"""Restore, prepare, run or analyze the four experiments used by the paper."""
import argparse,json
from pathlib import Path
from paper.common import prepare,execute
from paper.analysis import analyze

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['restore','prepare','run','analyze'])
    p.add_argument('experiment',choices=['e0','e1','e2','batching'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=1,help='preparation only; validation is serial; parallel E2 requires calibration')
    p.add_argument('--calibration',type=Path)
    p.add_argument('--resume',action='store_true',help='resume verified complete E2 stages only')
    args=p.parse_args();root=args.output.resolve()
    if args.action=='restore':
        if args.resume:p.error('restore requires a new output directory')
        from paper.restore import restore
        result=restore(args.experiment,root)
    elif args.action=='prepare':
        if args.resume:p.error('prepare requires a new output directory')
        result=prepare(args.experiment,root,args.concurrency,args.calibration)
    elif args.action=='run':execute(args.experiment,root,args.resume);result=analyze(args.experiment,root)
    else:result=analyze(args.experiment,root)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
