"""Pinned third-party AdaIN qualification. Downloads once, no training/test images."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

COMMIT='47950d0e6656a95a80a4b105c4c0f58d38ef785c'
REPO='https://github.com/naoto0804/pytorch-AdaIN'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('.cache/adain_qualified'))
    p.add_argument('--output',type=Path,default=Path('outputs/adain_qualification_v1'))
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a new output folder; existing results are preserved')
    args.cache.mkdir(parents=True,exist_ok=True)
    urls={name:f'https://raw.githubusercontent.com/naoto0804/pytorch-AdaIN/{COMMIT}/{name}' for name in ('net.py','function.py','LICENSE','README.md','test.py')}
    urls.update({name:f'{REPO}/releases/download/v0.0.0/{name}' for name in ('decoder.pth','vgg_normalised.pth')})
    lock=args.cache/'sources.json'
    old=json.loads(lock.read_text()) if lock.exists() else {}
    sources={}
    for name,url in urls.items():
        path=args.cache/name
        if not path.exists():
            print('Downloading:',name,flush=True)
            request=urllib.request.Request(url,headers={'User-Agent':'AdaIN-baseline-qualification'})
            with urllib.request.urlopen(request,timeout=90) as response, path.with_suffix(path.suffix+'.part').open('wb') as f:
                import shutil
                shutil.copyfileobj(response,f)
            path.with_suffix(path.suffix+'.part').replace(path)
        digest=sha(path)
        if name in old and old[name]['sha256']!=digest: raise ValueError('Cached file changed: '+name)
        sources[name]={'url':url,'sha256':digest,'bytes':path.stat().st_size}
    lock.write_text(json.dumps(sources,indent=2),encoding='utf-8')
    # Source is pinned above; weights must load as state dictionaries, not arbitrary objects.
    sys.path.insert(0,str(args.cache.resolve()))
    import torch
    from PIL import Image,ImageDraw
    from torchvision.transforms.functional import to_pil_image
    import net
    from function import adaptive_instance_normalization
    from stylization.external_split_data import ExternalManifestDataset
    from stylization.reference_training_data import validate_selection
    torch.set_num_threads(4)
    net.vgg.load_state_dict(torch.load(args.cache/'vgg_normalised.pth',map_location='cpu',weights_only=True),strict=True)
    net.decoder.load_state_dict(torch.load(args.cache/'decoder.pth',map_location='cpu',weights_only=True),strict=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder=torch.nn.Sequential(*list(net.vgg.children())[:31]).to(device).eval()
    decoder=net.decoder.to(device).eval()
    config=json.loads((args.root/'configs/reference_smoke_train.json').read_text(encoding='utf-8-sig'))
    manifest=args.root/config['manifest']
    if sha(manifest)!=config['manifest_sha256']: raise ValueError('Manifest changed')
    validate_selection(config,json.loads(manifest.read_text(encoding='utf-8-sig')))
    dataset=ExternalManifestDataset(manifest,args.data_root,'validation',256,config['seed'])
    indices=config['selection']['validation_pair_indices']
    args.output.mkdir(parents=True)
    alphas=[0.,.25,.5,.75,1.]
    records=[]
    with torch.inference_mode():
        for group in range(4):
            grid=Image.new('RGB',(256*7,280*2),'white'); draw=ImageDraw.Draw(grid)
            full,zero=[],[]
            for row,index in enumerate(indices[group*2:group*2+2]):
                sample=dataset[index]
                content=((sample['content']+1)/2).unsqueeze(0).to(device)
                reference=((sample['style']+1)/2).unsqueeze(0).to(device)
                cf,sf=encoder(content),encoder(reference)
                target=adaptive_instance_normalization(cf,sf)
                torch.testing.assert_close(target.mean((2,3)),sf.mean((2,3)),atol=1e-4,rtol=1e-4)
                outputs=[decoder(torch.lerp(cf,target,a)) for a in alphas]
                if any(not torch.isfinite(t).all() for t in outputs): raise FloatingPointError('Nonfinite inference')
                torch.testing.assert_close(outputs[0],decoder(cf),atol=1e-6,rtol=1e-5)
                full.append(outputs[-1]); zero.append(outputs[0])
                for col,(label,t) in enumerate(zip(['Content','Reference',*[f'Feature alpha {a:g}' for a in alphas]],[content,reference,*outputs])):
                    draw.text((col*256+4,row*280+3),label,fill='black')
                    grid.paste(to_pil_image(t[0].cpu().clamp(0,1)),(col*256,row*280+24))
                records.append({'validation_index':index,'content':sample['content_path'],'reference':sample['style_path'],
                                'reconstruction_mae_rgb01':(outputs[0]-content).abs().mean().item(),
                                'clipped_fraction_alpha1':((outputs[-1]<0)|(outputs[-1]>1)).float().mean().item()})
            torch.testing.assert_close(zero[0],zero[1],atol=1e-6,rtol=1e-5)
            grid.save(args.output/f'{group+1:03d}_comparison.png')
            print(f'Grid {group+1}/4; reference output difference RGB[0,1]={(full[0]-full[1]).abs().mean().item():.6f}',flush=True)
    report={'method':'AdaIN, Huang and Belongie ICCV 2017','implementation':'naoto0804/pytorch-AdaIN (unofficial PyTorch port)',
            'commit':COMMIT,'sources':sources,'manifest_sha256':sha(manifest),'split':'validation','image_size':256,
            'preprocessing':'EXIF RGB, square resize 256, float [0,1]; NO ImageNet mean/std normalization',
            'device':str(device),'torch_version':str(torch.__version__),'alphas':alphas,'records':records,
            'limitations':'Alpha zero is decoder reconstruction, not pixel identity. Clamping only for display. Pretrained data overlap with current art dataset not excluded; this is qualification, not held-out generalization. Upstream MIT code license retained; do not infer a blanket license for training images or pretrained weights. No training performed.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('QUALIFICATION INFERENCE COMPLETE:',args.output.resolve(),flush=True)


if __name__=='__main__': main()
