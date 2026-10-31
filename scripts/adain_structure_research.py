"""AdaIN structure hypothesis: Gaussian-pyramid normalized edge descriptors."""
import torch
import torch.nn.functional as F


def reduce_scale(x):
    kernel=x.new_tensor([1,4,6,4,1])/16
    kernel=(kernel[:,None]*kernel[None,:]).view(1,1,5,5)
    return F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),kernel,stride=2)


def descriptor(x):
    k=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/8
    gradient=F.conv2d(x,torch.stack((k,k.T)).unsqueeze(1))
    magnitude=(gradient.square().sum(1,keepdim=True)+1e-12).sqrt()
    return gradient/(magnitude.square()+.01**2).sqrt(), magnitude


def structure_loss(generated,content,mode):
    if mode not in ('none','single','multi'): raise ValueError('Unknown mode')
    if generated.shape!=content.shape or generated.ndim!=4 or generated.shape[1]!=3:
        raise ValueError('Expected matching NCHW RGB tensors')
    if mode=='none': return generated.sum()*0
    with torch.autocast(device_type=generated.device.type,enabled=False):
        weights=generated.new_tensor([.299,.587,.114],dtype=torch.float32).view(1,3,1,1)
        g=(generated.float()*weights).sum(1,keepdim=True)
        c=(content.detach().float()*weights).sum(1,keepdim=True)
        values=[]
        for level in range(1 if mode=='single' else 3):
            if level: g,c=reduce_scale(g),reduce_scale(c)
            gd,_=descriptor(g); cd,cm=descriptor(c)
            edge_weight=cm/(cm+.01)
            error=(gd-cd).square().mean(1,keepdim=True)
            values.append(((error*edge_weight).sum((1,2,3))/edge_weight.sum((1,2,3)).clamp_min(1e-6)).mean())
        if mode=='single': return values[0]
        return sum(w*v for w,v in zip((.2,.3,.5),values))
