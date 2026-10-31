"""Frozen multi-seed confirmation worker; validation only."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/adain_screen_v1'))
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--seed',type=int,default=17)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output directory; existing evidence is preserved')
    if args.smoke and args.output.name=='adain_screen_v1': raise ValueError('Use separate smoke directory')
    import torch
    import torch.nn.functional as F
    from PIL import Image,ImageDraw
    from torchvision.transforms.functional import to_pil_image
    from stylization.external_split_data import ExternalManifestDataset
    from adain_structure_research import structure_loss
    cache=args.root/'.cache/adain_qualified'
    provenance=json.loads((cache/'sources.json').read_text())
    for name,record in provenance.items():
        if digest(cache/name)!=record['sha256']: raise ValueError('Upstream asset changed: '+name)
    sys.path.insert(0,str(cache.resolve()))
    import net
    from function import adaptive_instance_normalization,calc_mean_std
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    manifest_path=args.root/'configs/vangogh_split_v1.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    def choose(records,count,label):
        order=sorted(range(len(records)),key=lambda i:hashlib.sha256(('42:adain:'+label+':'+records[i]['pixel_sha256']).encode()).hexdigest())
        result,groups=[],set()
        for i in order:
            if records[i]['group_id'] not in groups:
                result.append(i);groups.add(records[i]['group_id'])
            if len(result)==count:return result
        raise ValueError('Insufficient groups')
    size,steps=(64,2) if args.smoke else (256,500)
    train_ids=choose(manifest['splits']['train']['content'],128,'train')
    contents=choose(manifest['splits']['validation']['content'],1 if args.smoke else 16,'validation')
    styles=choose(manifest['splits']['validation']['style'],2 if args.smoke else 4,'styles')
    # Pair schedule is generated once and shared by all three arms.
    rng=random.Random(args.seed)
    schedule=[(rng.choice(train_ids),rng.randrange(len(manifest['splits']['train']['style']))) for _ in range(steps)]
    train=ExternalManifestDataset(manifest_path,args.data_root,'train',size,42)
    val=ExternalManifestDataset(manifest_path,args.data_root,'validation',size,42)
    net.vgg.load_state_dict(torch.load(cache/'vgg_normalised.pth',map_location='cpu',weights_only=True),strict=True)
    net.decoder.load_state_dict(torch.load(cache/'decoder.pth',map_location='cpu',weights_only=True),strict=True)
    layers=list(net.vgg.children())
    encoder=torch.nn.Sequential(*layers[:31]).to(device).eval().requires_grad_(False)
    initial=copy.deepcopy(net.decoder.state_dict())
    def encode(x):
        outputs=[]
        for i,layer in enumerate(encoder):
            x=layer(x)
            if i in (3,10,17,30):outputs.append(x)
        return outputs
    def style_error(left,right):
        return sum(F.mse_loss(a,b) for g,s in zip(left,right) for a,b in zip(calc_mean_std(g),calc_mean_std(s)))
    plan={'phase':'wiring_smoke' if args.smoke else 'three_seed_validation_confirmation',
          'seed':args.seed,'steps':steps,'size':size,'lr':1e-5,'adam_betas':[.9,.999],
          'content_weight':1,'style_weight':10,'structure_weight':1,
          'structure':'normalized Sobel descriptors on Gaussian pyramid; multi weights .2/.3/.5, epsilon .01',
          'train_schedule':schedule,'validation_contents':contents,'validation_styles':styles,
          'manifest_sha256':digest(manifest_path),'upstream':provenance,
          'local_code_sha256':{Path(__file__).name:digest(Path(__file__)),'adain_structure_research.py':digest(Path(__file__).with_name('adain_structure_research.py'))},
          'limitations':'No test images. Pretraining overlap unknown. One worker of a three-seed validation study; not an untouched test evaluation. Metrics partly share the training encoder. No automatic winner. Alpha is feature interpolation.'}
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    rows=[]
    alphas=[0.,.25,.5,.75,1.]
    def evaluate(decoder,arm):
        decoder.eval()
        grid=Image.new('RGB',(size*7,(size+20)*len(styles)),'white');draw=ImageDraw.Draw(grid)
        with torch.inference_mode():
            for ci,c in enumerate(contents):
                for si,s in enumerate(styles):
                    index=c*len(val.style_files)+s
                    sample=val[index]
                    content=((sample['content']+1)/2).unsqueeze(0).to(device)
                    reference=((sample['style']+1)/2).unsqueeze(0).to(device)
                    cf,sf=encode(content),encode(reference)
                    target=adaptive_instance_normalization(cf[-1],sf[-1])
                    pictures=[content,reference]
                    for alpha in alphas:
                        result=decoder(torch.lerp(cf[-1],target,alpha))
                        rf=encode(result)
                        row={'arm':arm,'pair_index':index,'alpha':alpha,
                             'style':style_error(rf,sf).item(),
                             'content_feature':F.mse_loss(rf[-1],cf[-1]).item(),
                             'pixel_l1':F.l1_loss(result,content).item(),
                             'pooled_intensity_l1':F.l1_loss(F.avg_pool2d(result.mean(1,keepdim=True),16),F.avg_pool2d(content.mean(1,keepdim=True),16)).item(),
                             'single_structure':structure_loss(result,content,'single').item(),
                             'multi_structure':structure_loss(result,content,'multi').item(),
                             'clipped_fraction':((result<0)|(result>1)).float().mean().item()}
                        if not all(torch.isfinite(torch.tensor(v)) for k,v in row.items() if k not in ('arm','pair_index')):raise FloatingPointError('Nonfinite metrics')
                        rows.append(row);pictures.append(result)
                    if ci==0:
                        for col,(label,t) in enumerate(zip(['Content','Reference',*[f'alpha {a:g}' for a in alphas]],pictures)):
                            draw.text((col*size+2,si*(size+20)+2),label,fill='black')
                            grid.paste(to_pil_image(t[0].cpu().clamp(0,1)),(col*size,si*(size+20)+20))
        grid.save(args.output/f'{arm}_curve.png')
        with (args.output/'per_pair_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('Evaluated:',arm,flush=True)
    decoder=copy.deepcopy(net.decoder).to(device)
    evaluate(decoder,'pretrained')
    for mode in ('none','single','multi'):
        torch.manual_seed(args.seed)
        decoder.load_state_dict(initial,strict=True);decoder.train()
        optimizer=torch.optim.Adam(decoder.parameters(),lr=1e-5)
        started=time.monotonic();history=[]
        for step,(c,s) in enumerate(schedule,1):
            content=((train.read_image(train.content_files[c])+1)/2).unsqueeze(0).to(device)
            reference=((train.read_image(train.style_files[s])+1)/2).unsqueeze(0).to(device)
            with torch.no_grad():
                cf,sf=encode(content),encode(reference)
                target=adaptive_instance_normalization(cf[-1],sf[-1])
            optimizer.zero_grad(set_to_none=True)
            result=decoder(target);rf=encode(result)
            lc=F.mse_loss(rf[-1],target);ls=style_error(rf,sf);le=structure_loss(result,content,mode)
            loss=lc+10*ls+le
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in decoder.parameters()):raise FloatingPointError('Invalid gradients')
            optimizer.step()
            history.append({'step':step,'content':lc.item(),'style':ls.item(),'structure':le.item()})
            if step%50==0 or step==steps:print(f'{mode} {step}/{steps} elapsed={time.monotonic()-started:.1f}s',flush=True)
        torch.save({'decoder':decoder.cpu().state_dict(),'mode':mode,'plan':plan,'steps':steps},args.output/f'{mode}_decoder.pt')
        decoder.to(device)
        (args.output/f'{mode}_training.json').write_text(json.dumps(history),encoding='utf-8')
        evaluate(decoder,mode)
    means=[]
    for arm in ('pretrained','none','single','multi'):
        for alpha in alphas:
            subset=[r for r in rows if r['arm']==arm and r['alpha']==alpha]
            means.append({'arm':arm,'alpha':alpha,**{k:sum(r[k] for r in subset)/len(subset) for k in ('style','content_feature','pixel_l1','pooled_intensity_l1','single_structure','multi_structure','clipped_fraction')}})
    (args.output/'curve_summary.json').write_text(json.dumps(means,indent=2),encoding='utf-8')
    (args.output/'COMPLETE.json').write_text(json.dumps({'seed':args.seed,'rows':len(rows),'smoke':args.smoke}),encoding='utf-8')
    print('CONFIRMATION WORKER COMPLETE:',args.output.resolve(),flush=True)
    print('Compare overlapping style-error ranges; do not rank structure at unmatched style strength.',flush=True)


if __name__=='__main__':main()
