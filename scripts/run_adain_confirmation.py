"""Run the frozen three-seed study in one command, then summarize it."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,default=Path('outputs/adain_confirmation_v1'))
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke and a.output.name=='adain_confirmation_v1':raise ValueError('Smoke needs a separate output directory')
    here=Path(__file__).resolve().parent
    worker=here/'run_adain_confirm_worker.py'
    seeds=[17,43,101] # New schedules, excluding the screening seed 42.
    locked={'seeds':seeds,'smoke':a.smoke,'data_root':str(a.data_root.resolve()),'root':str(a.root.resolve()),
            'files':{name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ('run_adain_confirm_worker.py','adain_structure_research.py','summarize_adain_confirmation.py')},
            'manifest':hashlib.sha256((a.root/'configs/vangogh_split_v1.json').read_bytes()).hexdigest(),
            'assets':hashlib.sha256((a.root/'.cache/adain_qualified/sources.json').read_bytes()).hexdigest()}
    a.output.mkdir(parents=True,exist_ok=True)
    lock=a.output/'protocol.json'
    if lock.exists():
        if json.loads(lock.read_text())!=locked:raise ValueError('Protocol changed; use a new output folder')
    else:lock.write_text(json.dumps(locked,indent=2),encoding='utf-8')
    for seed in seeds:
        target=a.output/f'seed_{seed}'
        if (target/'COMPLETE.json').exists():
            print('Keeping completed seed:',seed,flush=True);continue
        if target.exists():
            # Preserve interrupted evidence; restart this seed from the same pretrained weights.
            n=1
            while (a.output/f'seed_{seed}_interrupted_{n}').exists():n+=1
            target.rename(a.output/f'seed_{seed}_interrupted_{n}')
        command=[sys.executable,'-B',str(worker),'--root',str(a.root.resolve()),'--data-root',str(a.data_root.resolve()),'--output',str(target.resolve()),'--seed',str(seed)]
        if a.smoke:command.append('--smoke')
        subprocess.run(command,check=True)
    subprocess.run([sys.executable,'-B',str(here/'summarize_adain_confirmation.py'),str(a.output.resolve())],check=True)
    print('ALL THREE SEEDS COMPLETE:',a.output.resolve(),flush=True)

if __name__=='__main__':main()
