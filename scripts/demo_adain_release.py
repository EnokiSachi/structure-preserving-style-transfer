"""Local research demo backed by the frozen multi-scale AdaIN checkpoint."""
import argparse
from datetime import datetime
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from PIL import Image,ImageTk,ImageOps
from adain_final_inference import FrozenStylizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    root=a.root
    w=tk.Tk();w.title('可控图像风格化 · 多尺度结构研究');w.geometry('1090x690')
    f=ttk.Frame(w,padding=20);f.pack(fill='both',expand=True)
    ttk.Label(f,text='参考图像驱动的风格迁移',font=('Microsoft YaHei UI',20,'bold')).pack(anchor='w')
    ttk.Label(f,text='本地运行 · 256 × 256 输出 · 内容图和参考图会缩放为正方形').pack(anchor='w',pady=8)
    status=tk.StringVar(value='正在载入冻结模型……');strength=tk.DoubleVar(value=.5)
    paths={};photos={};state={'model':None,'busy':False,'result':None,'timer':None,'prepared':None};q=queue.Queue()
    bar=ttk.Frame(f);bar.pack(fill='x')
    panels=ttk.Frame(f);panels.pack(fill='x',pady=18)
    labels={}
    for key,title in [('content','内容图'),('reference','参考画作'),('result','生成结果')]:
        col=ttk.Frame(panels);col.pack(side='left',expand=True)
        ttk.Label(col,text=title).pack();labels[key]=ttk.Label(col);labels[key].pack()
    def display(key,image):
        photos[key]=ImageTk.PhotoImage(image.resize((300,300)))
        labels[key].configure(image=photos[key])
    def controls():
        ready=state['model'] is not None and not state['busy']
        generate.configure(state='normal' if ready and len(paths)==2 else 'disabled')
        save.configure(state='normal' if ready and state['result'] is not None else 'disabled')
        slider.configure(state='normal' if ready else 'disabled')
        for b in selectors:b.configure(state='normal' if not state['busy'] else 'disabled')
    def choose(key):
        filename=filedialog.askopenfilename(filetypes=[('图像','*.png *.jpg *.jpeg *.webp *.bmp')])
        if not filename:return
        try:
            with Image.open(filename) as im:preview=ImageOps.exif_transpose(im).convert('RGB').resize((256,256))
        except Exception as e:messagebox.showerror('无法读取图像',str(e));return
        paths[key]=filename;display(key,preview);state['prepared']=None;state['result']=None
        labels['result'].configure(image='');status.set('选择好两张图后，点击生成。');controls()
    selectors=[]
    for key,title in [('content','选择内容图'),('reference','选择参考图')]:
        b=ttk.Button(bar,text=title,command=lambda k=key:choose(k));b.pack(side='left',padx=4);selectors.append(b)
    def run():
        if state['busy'] or state['model'] is None or len(paths)!=2:return
        if state['timer'] is not None:w.after_cancel(state['timer']);state['timer']=None
        snapshot=dict(paths);alpha=float(strength.get());state['busy']=True;state['result']=None;controls();status.set('正在生成……')
        def work():
            try:
                model=state['model'];images=None
                if state['prepared']!=snapshot:images=model.prepare(snapshot['content'],snapshot['reference'])
                q.put(('result',(snapshot,alpha,images,model.render(alpha))))
            except Exception as e:q.put(('error',str(e)))
        threading.Thread(target=work,daemon=True).start()
    generate=ttk.Button(bar,text='生成',command=run,state='disabled');generate.pack(side='left',padx=8)
    def save_result():
        if state['result'] is None:return
        folder=root/'outputs/adain_exports';folder.mkdir(parents=True,exist_ok=True)
        name=filedialog.asksaveasfilename(initialdir=str(folder),initialfile='stylized_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.png',defaultextension='.png',filetypes=[('PNG','*.png')])
        if name:
            try:state['result'].save(name,format='PNG');status.set('已保存：'+str(Path(name).resolve()))
            except Exception as e:messagebox.showerror('保存失败',str(e))
    save=ttk.Button(bar,text='保存 PNG',command=save_result,state='disabled');save.pack(side='left',padx=4)
    ttk.Label(f,text='风格强度：0 为内容重建，1 为完整风格变换；拖动后自动重新生成。').pack(anchor='w',pady=8)
    def changed(_):
        if state['timer'] is not None:w.after_cancel(state['timer'])
        if state['result'] is not None:
            # Do not allow exporting a stale strength while a new render is pending.
            save.configure(state='disabled');state['timer']=w.after(250,run)
    slider=ttk.Scale(f,from_=0,to=1,variable=strength,command=changed,state='disabled');slider.pack(fill='x')
    ttk.Label(f,textvariable=status,wraplength=1030).pack(anchor='w',pady=14)
    ttk.Label(f,text='高强度可能改变轮廓与颜色。默认保存目录：outputs/adain_exports；保存后会显示完整路径。',wraplength=1030).pack(anchor='w')
    def poll():
        try:
            while True:
                kind,value=q.get_nowait()
                if kind=='model':state['model']=value;status.set('模型就绪。请选择内容图与参考画作。')
                elif kind=='result':
                    snapshot,alpha,images,result=value
                    state['prepared']=snapshot;state['result']=result;state['busy']=False
                    if images:
                        display('content',images[0]);display('reference',images[1])
                    display('result',result);status.set(f'已生成 · 强度 {alpha:.2f}。点击“保存 PNG”导出。')
                else:
                    state['busy']=False;status.set('操作失败：'+value);messagebox.showerror('操作失败',value)
                controls()
        except queue.Empty:pass
        w.after(100,poll)
    def load():
        try:q.put(('model',FrozenStylizer(root)))
        except Exception as e:q.put(('error',str(e)))
    threading.Thread(target=load,daemon=True).start();poll();w.mainloop()

if __name__=='__main__':main()
