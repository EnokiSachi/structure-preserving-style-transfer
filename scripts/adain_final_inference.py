"""Frozen offline AdaIN inference; feature interpolation, not image blending."""
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

class FrozenStylizer:
    def __init__(self,root,lock_path=None):
        import torch
        self.torch=torch
        self.root=Path(root)
        self.lock=json.loads(Path(lock_path or self.root/'configs/adain_release_v1.json').read_text(encoding='utf-8'))
        for name,digest in self.lock['runtime_files'].items():
            if sha(self.root/name)!=digest:raise ValueError('Frozen asset changed: '+name)
        cache=self.root/'.cache/adain_qualified'
        sys.path.insert(0,str(cache.resolve()))
        import net
        from function import adaptive_instance_normalization
        self.adain=adaptive_instance_normalization
        torch.set_num_threads(4)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.vgg.load_state_dict(torch.load(cache/'vgg_normalised.pth',map_location='cpu',weights_only=True),strict=True)
        self.encoder=torch.nn.Sequential(*list(net.vgg.children())[:31]).eval().to(self.device)
        checkpoint=torch.load(self.root/self.lock['checkpoint'],map_location='cpu',weights_only=True)
        if checkpoint['mode']!='multi' or checkpoint['steps']!=500 or checkpoint['plan']['seed']!=17:
            raise ValueError('Unexpected frozen checkpoint metadata')
        net.decoder.load_state_dict(checkpoint['decoder'],strict=True)
        self.decoder=net.decoder.eval().to(self.device)
        self.features=None

    def prepare(self,content_path,reference_path):
        from PIL import Image,ImageOps
        from torchvision.transforms.functional import to_tensor
        images=[];tensors=[]
        for path in (content_path,reference_path):
            with Image.open(path) as source:
                im=ImageOps.exif_transpose(source).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
            images.append(im);tensors.append(to_tensor(im).unsqueeze(0).to(self.device))
        with self.torch.inference_mode():
            cf=self.encoder(tensors[0]);sf=self.encoder(tensors[1])
            self.features=(cf,self.adain(cf,sf))
        return images

    def render(self,alpha):
        from torchvision.transforms.functional import to_pil_image
        if self.features is None:raise ValueError('Choose both images first')
        if not 0<=alpha<=1:raise ValueError('Strength must be between zero and one')
        with self.torch.inference_mode():
            output=self.decoder(self.torch.lerp(*self.features,float(alpha)))
            if not self.torch.isfinite(output).all():raise FloatingPointError('Nonfinite output')
        return to_pil_image(output[0].cpu().clamp(0,1))

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--content',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--strength',type=float,default=.5)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Output already exists')
    model=FrozenStylizer(a.root);model.prepare(a.content,a.reference)
    im=model.render(a.strength);a.output.parent.mkdir(parents=True,exist_ok=True);im.save(a.output,format='PNG')
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
