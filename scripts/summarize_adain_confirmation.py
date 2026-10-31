"""Paired interpolation diagnostics; no claim that seeds or pairs are independent replications."""
import csv
import json
import statistics
import sys
from pathlib import Path

METRICS=('multi_structure','single_structure','content_feature','pixel_l1','pooled_intensity_l1')
ARMS=('pretrained','none','single','multi')

def interpolate(curve,target,key):
    curve=sorted(curve,key=lambda r:r['alpha'])
    if any(b['style']>=a['style'] for a,b in zip(curve,curve[1:])):return None
    for a,b in zip(curve,curve[1:]):
        if b['style']<=target<=a['style']:
            t=(target-a['style'])/(b['style']-a['style'])
            return a[key]+t*(b[key]-a[key])
    return None # Never extrapolate or force a nonmonotonic curve to match.

def main():
    root=Path(sys.argv[1]);protocol=json.loads((root/'protocol.json').read_text())
    matched=[];seed_means=[];coverage=[]
    for seed in protocol['seeds']:
        with (root/f'seed_{seed}'/'per_pair_metrics.csv').open(encoding='utf-8') as f:
            rows=[{k:(v if k=='arm' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
        pairs=sorted({r['pair_index'] for r in rows})
        accepted=0
        for pair in pairs:
            curves={arm:[r for r in rows if r['arm']==arm and r['pair_index']==pair] for arm in ARMS}
            for alpha in (.25,.5,.75):
                target=next(r['style'] for r in curves['none'] if r['alpha']==alpha)
                values={arm:{k:interpolate(curves[arm],target,k) for k in METRICS} for arm in ARMS}
                if any(v is None for d in values.values() for v in d.values()):continue
                accepted+=1
                for arm in ARMS:
                    matched.append({'seed':seed,'pair_index':int(pair),'none_alpha':alpha,'matched_style':target,'arm':arm,**values[arm]})
        coverage.append({'seed':seed,'accepted':accepted,'possible':len(pairs)*3})
        for arm in ARMS:
            group=[r for r in matched if r['seed']==seed and r['arm']==arm]
            if group:seed_means.append({'seed':seed,'arm':arm,'matched_points':len(group),**{k:statistics.mean(r[k] for r in group) for k in METRICS}})
    if matched:
        with (root/'matched_pairs.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(matched[0]));w.writeheader();w.writerows(matched)
    (root/'matched_summary.json').write_text(json.dumps({'coverage':coverage,'seed_means':seed_means},indent=2),encoding='utf-8')
    lines=['# Frozen AdaIN confirmation results','',f"Smoke mode: {protocol['smoke']}",'',
      'Validation only. Linear interpolation at the no-structure arm\'s per-pair style errors at alpha 0.25/0.5/0.75. All four arms must overlap and have strictly decreasing style error; unmatched or nonmonotonic curves are excluded. No extrapolation. This is an approximation from five strengths, not newly generated style-matched outputs.', '',
      'Seeds vary training order and reference pairings; all start from identical pretrained weights. Images and styles are reused across seeds. Standard deviations below describe three training schedules, not independent datasets or statistical significance.', '',
      '| Seed | Matched points / possible |','|---|---|']
    lines += [f"| {r['seed']} | {r['accepted']} / {r['possible']} |" for r in coverage]
    lines += ['','| Arm | Metric | Mean across seeds | SD across seeds |','|---|---|---:|---:|']
    for arm in ARMS:
        for k in METRICS:
            vals=[r[k] for r in seed_means if r['arm']==arm]
            if vals:lines.append(f"| {arm} | {k} | {statistics.mean(vals):.6f} | {statistics.stdev(vals) if len(vals)>1 else 0:.6f} |")
    lines += ['','Structure metrics and VGG metrics share the training objective or encoder. Pooled intensity L1 is an additional non-trained diagnostic, but is sensitive to legitimate color changes and is not perceptual ground truth. Pretraining overlap is unknown. Expanded validation includes the screening pairs. No automatic winner, test claim, novelty claim, or CV improvement percentage is authorized by this report.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved matched-pair analysis and RESULTS.md',flush=True)

if __name__=='__main__':main()
