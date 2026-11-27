"""ML Lab 7 complete experiment runner. Python 3.10+, CPU or GPU training; CPU-only benchmarks."""
import argparse, copy, csv, gzip, io, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def loaders(cfg):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    data = ROOT / 'data'; data.mkdir(exist_ok=True)
    tr = datasets.FashionMNIST(str(data), train=True, download=True, transform=tf)
    te = datasets.FashionMNIST(str(data), train=False, download=True, transform=tf)
    nval=6000; ntrain=len(tr)-nval
    train, val = random_split(tr,[ntrain,nval],generator=torch.Generator().manual_seed(cfg['seed']))
    opt=dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], pin_memory=torch.cuda.is_available())
    return (DataLoader(train,shuffle=True,**opt),DataLoader(val,shuffle=False,**opt),DataLoader(te,shuffle=False,**opt))

class CNN(nn.Module):
    def __init__(self,channels=(32,64,128),hidden=160):
        super().__init__()
        a,b,c=channels
        self.channels=tuple(channels); self.hidden=hidden
        self.conv1=nn.Conv2d(1,a,3,padding=1)
        self.conv2=nn.Conv2d(a,b,3,padding=1)
        self.conv3=nn.Conv2d(b,c,3,padding=1)
        self.fc1=nn.Linear(c*7*7,hidden)
        self.fc2=nn.Linear(hidden,10)
    def forward(self,x):
        x=F.relu(self.conv1(x)); x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=F.max_pool2d(F.relu(self.conv3(x)),2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

def params(model): return sum(p.numel() for p in model.parameters())

def evaluate(model,loader,device):
    model.eval(); good=total=0; loss_sum=0.
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            out=model(x)
            loss_sum += F.cross_entropy(out,y,reduction='sum').item()
            good += (out.argmax(1)==y).sum().item(); total+=len(y)
    return {'accuracy':100*good/total,'loss':loss_sum/total}

def train(model,train_loader,val_loader,device,epochs,lr=0.001,teacher=None,alpha=.5,T=4.,masks=None,tag=''):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    best_acc=-1.; best=None; history=[]
    if teacher is not None: teacher.eval()
    for epoch in range(1,epochs+1):
        model.train(); n=0; total_loss=0.
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.cross_entropy(logits,y)
            if teacher is not None:
                with torch.no_grad(): tlog=teacher(x)
                # Conventional teacher-to-student KL: KL(P_teacher || P_student).
                kd=F.kl_div(F.log_softmax(logits/T,dim=1),F.softmax(tlog/T,dim=1),reduction='batchmean')
                loss=alpha*loss+(1-alpha)*(T*T)*kd
            loss.backward()
            if masks:
                for name,p in model.named_parameters():
                    if name in masks: p.grad.mul_(masks[name])
            optimizer.step()
            if masks:
                with torch.no_grad():
                    for name,p in model.named_parameters():
                        if name in masks: p.mul_(masks[name])
            n+=len(y); total_loss+=loss.item()*len(y)
        ev=evaluate(model,val_loader,device)
        history.append({'epoch':epoch,'train_loss':total_loss/n,'val_accuracy':ev['accuracy']})
        if ev['accuracy']>best_acc:
            best_acc=ev['accuracy']; best=copy.deepcopy(model.state_dict())
        print(f'{tag} epoch {epoch}/{epochs} loss={total_loss/n:.4f} val_acc={ev["accuracy"]:.2f}%',flush=True)
    model.load_state_dict(best)
    return history

def unstructured_mask(model,sparsity,mode):
    # Prune eligible weights only, not bias; zero eligible weight ratio is computed below.
    items=[(n,p) for n,p in model.named_parameters() if n.endswith('weight')]
    masks={}
    if mode=='global':
        allweights=torch.cat([p.detach().abs().flatten() for _,p in items])
        k=min(int(allweights.numel()*sparsity),allweights.numel())
        if k==0: return {n:torch.ones_like(p) for n,p in items}
        # exact k positions, stable ties handled by torch.argsort
        flatmask=torch.ones_like(allweights)
        flatmask[torch.argsort(allweights)[:k]]=0.
        pos=0
        for n,p in items:
            count=p.numel(); masks[n]=flatmask[pos:pos+count].reshape_as(p);pos+=count
    else:
        for n,p in items:
            a=p.detach().abs().flatten();k=int(a.numel()*sparsity)
            m=torch.ones_like(a)
            if k:m[torch.argsort(a)[:k]]=0.
            masks[n]=m.reshape_as(p)
    return masks

def apply_masks(model,masks):
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in masks:p.mul_(masks[n])

def structured_channel_prune(original,ratio):
    # Physically rebuild convolutional layers and FC input to yield genuine dense CPU speedups.
    orig=copy.deepcopy(original).cpu().eval()
    convs=[orig.conv1,orig.conv2,orig.conv3]
    kept=[]
    for conv in convs:
        norms=conv.weight.detach().flatten(1).norm(p=2,dim=1)
        keep=max(1,round(conv.out_channels*(1-ratio)))
        kept.append(torch.argsort(norms,descending=True)[:keep].sort().values)
    new=CNN(tuple(len(x) for x in kept), orig.hidden)
    with torch.no_grad():
        k1,k2,k3=kept
        new.conv1.weight.copy_(orig.conv1.weight[k1]);new.conv1.bias.copy_(orig.conv1.bias[k1])
        new.conv2.weight.copy_(orig.conv2.weight[k2][:,k1]);new.conv2.bias.copy_(orig.conv2.bias[k2])
        new.conv3.weight.copy_(orig.conv3.weight[k3][:,k2]);new.conv3.bias.copy_(orig.conv3.bias[k3])
        # Flatten order: channels then spatial 7x7 pixels.
        fcw=orig.fc1.weight.reshape(orig.hidden,orig.channels[2],7,7)
        new.fc1.weight.copy_(fcw[:,k3,:,:].reshape(orig.hidden,-1))
        new.fc1.bias.copy_(orig.fc1.bias)
        new.fc2.load_state_dict(orig.fc2.state_dict())
    return new

def sizes(model):
    buf=io.BytesIO();torch.save(model.state_dict(),buf);dense=buf.getvalue()
    weights=np.concatenate([p.detach().cpu().numpy().reshape(-1) for n,p in model.named_parameters() if n.endswith('weight')])
    idx=np.flatnonzero(weights); values=weights[idx]
    # Explicit coordinate-index sparse representation, not a dense tensor with zeros.
    sparsebuf=io.BytesIO();np.savez_compressed(sparsebuf,indices=idx.astype(np.int32),values=values,
                                               shape=np.asarray([weights.size],dtype=np.int64))
    return {'checkpoint_mb':len(dense)/1e6,'checkpoint_gzip_mb':len(gzip.compress(dense))/1e6,
            'sparse_weights_npz_mb':len(sparsebuf.getvalue())/1e6,
            'weight_sparsity':1-len(idx)/max(len(weights),1)}

def latency(model,runs=100):
    model=copy.deepcopy(model).cpu().eval()
    torch.set_num_threads(min(4,max(os.cpu_count() or 1,1)))
    img=torch.rand((1,1,32,32))  # Benchmark includes resizing from 32x32 and normalization.
    def once():
        start=time.perf_counter()
        x=F.interpolate(img,size=(28,28),mode='bilinear',align_corners=False)
        x=(x-0.2860)/0.3530
        with torch.no_grad(): model(x)
        return (time.perf_counter()-start)*1000
    for _ in range(20):once()
    vals=np.array([once() for _ in range(runs)])
    return {'median_ms':float(np.median(vals)),'p95_ms':float(np.percentile(vals,95)),
            'median_images_per_sec':float(1000/np.median(vals))}

def macs(model):
    total=[0];hooks=[]
    def fn(mod,inp,out):
        if isinstance(mod,nn.Conv2d):
            total[0]+=out.shape[1]*out.shape[2]*out.shape[3]*(mod.in_channels//mod.groups)*mod.kernel_size[0]*mod.kernel_size[1]
        if isinstance(mod,nn.Linear):total[0]+=mod.in_features*mod.out_features
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):hooks.append(m.register_forward_hook(fn))
    with torch.no_grad():model.cpu()(torch.rand(1,1,28,28))
    for h in hooks:h.remove()
    return total[0]

def record(model,name,group,test_loader,cfg,sparsity=0.,extra=None):
    model=model.cpu().eval()
    ev=evaluate(model,test_loader,'cpu'); sz=sizes(model); t=latency(model,cfg['latency_runs'])
    result={'model':name,'group':group,'target_sparsity':sparsity,'params':params(model),'macs':macs(model),
            'test_accuracy_pct':ev['accuracy'],**sz,**t}
    if extra:result.update(extra)
    ck=ROOT/'checkpoints'/f'{name}.pt';torch.save(model.state_dict(),ck)
    return result

def write_rows(rows):
    dest=ROOT/'results'/'metrics.csv'
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(dest,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    with open(ROOT/'results'/'metrics.json','w') as f:json.dump(rows,f,indent=2)

def plots(rows):
    plt.figure(figsize=(8,5))
    for group in ('global','layerwise','structured'):
        x=[r['target_sparsity'] for r in rows if r['group']==group]
        y=[r['test_accuracy_pct'] for r in rows if r['group']==group]
        if x:plt.plot(x,y,marker='o',label=group)
    plt.xlabel('Target sparsity / fraction of filters removed');plt.ylabel('Test accuracy (%)')
    plt.title('Lab 7: Accuracy vs. pruning strength');plt.grid(alpha=.25);plt.legend();plt.tight_layout()
    plt.savefig(ROOT/'results'/'sparsity_curves.png',dpi=180);plt.close()
    chosen=[r for r in rows if r['group'] in ('teacher','student')]
    if chosen:
        fig,ax=plt.subplots(figsize=(7,4)); ax.bar([r['model'] for r in chosen],[r['test_accuracy_pct'] for r in chosen]);
        ax.set_ylim(0,100);ax.set_ylabel('Test accuracy (%)');ax.set_title('Plain student vs. distilled student');
        plt.xticks(rotation=15,ha='right');fig.tight_layout();fig.savefig(ROOT/'results'/'distillation_comparison.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.scatter([r['median_ms'] for r in chosen],[r['test_accuracy_pct'] for r in chosen],s=[max(25,r['checkpoint_mb']*30) for r in chosen])
        for r in chosen:ax.annotate(r['model'],(r['median_ms'],r['test_accuracy_pct']),xytext=(4,4),textcoords='offset points')
        ax.set_xlabel('Median CPU latency (ms; lower is better)');ax.set_ylabel('Test accuracy (%)')
        ax.set_title('Efficiency: bubble size proportional to model storage');fig.tight_layout()
        fig.savefig(ROOT/'results'/'accuracy_latency_size.png',dpi=180);plt.close(fig)

def make_report(rows,cfg,device):
    lines=['# Machine Learning Lab 7 — Compression: Pruning and Distillation','',
        '**Student:** Imam Hossain  **ID:** 73947  ','**Course:** Machine Learning (CII5SE039CI)','',
        '## Methodology and experimental setup','',
        f'- Dataset: Fashion-MNIST, 54,000 training / 6,000 validation / 10,000 test samples.',
        f'- Seed: {cfg["seed"]}; batch size: {cfg["batch_size"]}; initial training device: {device}.',
        f'- Teacher epochs: {cfg["teacher_epochs"]}; student epochs: {cfg["student_epochs"]}; mask-fixed fine-tune epochs: {cfg["finetune_epochs"]}.',
        '- All reported accuracies are on the official held-out Fashion-MNIST test set.',
        '- CPU latency is single-image batch=1 with 20 warmups, measured for the configured number of iterations.',
        '- Latency preprocessing represents tensor resize (32→28) and normalization, not file decoding or camera capture.',
        '- FLOPs are approximated as 2 × multiply-accumulate operations (MACs), omitting activations and pooling.',
        '- Energy: not measured (no power meter). CPU time and MACs are only proxies, not joules.',
        '- Unstructured sparse NPZ stores nonzero weight coordinates and values only, omitting biases and layer metadata; it is **not a deployable checkpoint**.',
        '', '## Results — all experiments','',
        '| Model | Test acc. % | Parameters | Sparsity | Checkpoint MB | Sparse weights NPZ MB | Median ms | p95 ms | MACs |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["test_accuracy_pct"]:.2f} | {r["params"]:,} | {r["weight_sparsity"]:.2%} | {r["checkpoint_mb"]:.3f} | {r["sparse_weights_npz_mb"]:.3f} | {r["median_ms"]:.2f} | {r["p95_ms"]:.2f} | {r["macs"]:,} |')
    lines += ['', '![Accuracy vs sparsity](results/sparsity_curves.png)','',
       '![Students compared](results/distillation_comparison.png)','',
       '![Latency / size tradeoff](results/accuracy_latency_size.png)','',
       '## Efficiency pillars (Chapter 9)','',
       '| Pillar | Evidence and interpretation |','|---|---|',
       '| Data | Same Fashion-MNIST train/validation/test splits for all experiments. No change in training data volume across models. |',
       '| Model | Compare parameter count, checkpoint size, nonzero weight sparsity and sparse representation. Sparse representation is not a runnable model file. |',
       '| Compute | Compare MACs and actual measured CPU latency. Masking unstructured weights does not change the dense computation graph. |',
       '| Energy | No joules measured: cannot claim energy savings. Reduced MACs and latency may suggest lower energy, but power draw is unknown. |',
       '', '## Task 3 — Honest latency and storage interpretation','',
       'Global and layer-wise magnitude masks retain original dense tensor shapes. Standard PyTorch dense convolutions and matrix multiplies still process all the zeros; latency is not expected to scale with sparsity. '
       'A sparse weight-only file can shrink storage at high sparsity but needs metadata and a sparse execution backend to run. '
       'Structured filter removal rebuilds smaller dense layers, reducing parameters and MACs; that may yield genuine hardware speedups, subject to CPU kernel efficiency.',
       '', '## Task 4 — Distillation','',
       f'- Temperature T={cfg["temperature"]}, alpha={cfg["alpha"]}.',
       '- Loss: alpha × cross-entropy with true labels + (1−alpha) × T² × KL(teacher probabilities || student probabilities).',
       '- A plain student uses the same small architecture and training length, without teacher logits.',
       '', '## Conclusion','']
    plain=next((r for r in rows if r['model']=='student_plain'),None)
    kd=next((r for r in rows if r['model']=='student_distilled'),None)
    teacher=next((r for r in rows if r['model']=='teacher'),None)
    if teacher and plain and kd:
        lines += [f'Teacher reached {teacher["test_accuracy_pct"]:.2f}% accuracy at {teacher["median_ms"]:.2f} ms median CPU latency. '
                  f'Plain student reached {plain["test_accuracy_pct"]:.2f}%; distilled student reached {kd["test_accuracy_pct"]:.2f}% '
                  f'(change {kd["test_accuracy_pct"]-plain["test_accuracy_pct"]:+.2f} percentage points).',
                  'Read the pruning table for the accuracy cost of increasing sparsity; do not assume that sparsity itself reduces dense-kernel latency.']
    else:lines += ['Experiment incomplete: do not report performance claims until all runs finish.']
    lines+=['', '## Limitations and reproducibility','',
            '- CPU benchmark numbers depend on processor, competing processes, and backend; repeat on the submission machine.',
            '- Each pruning run is initialized from the same teacher baseline; fine-tuning does not change target masks.',
            '- The teacher and student are selected by validation accuracy and evaluated on the test set.',
            '- Logs and checkpoints are saved under results/ and checkpoints/.',
            '- Task 2 pruning targets are applied to eligible convolutional and linear weights; biases are not masked.',
            '- For structured pruning, the stated fraction is the fraction of output filters removed from each convolution (rounded per layer).',
            '- Test set is evaluated for each configured experiment for educational reporting; no test-driven hyperparameter selection should be performed.',
            '']
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(ROOT/'config.yaml'));ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.smoke:
        cfg.update(teacher_epochs=1,student_epochs=1,finetune_epochs=1,latency_runs=10)
        cfg['sparsities']=[0,.5];cfg['structured_levels']=[.25]
    seed_everything(cfg['seed']); (ROOT/'results').mkdir(exist_ok=True);(ROOT/'checkpoints').mkdir(exist_ok=True)
    dev='cuda' if torch.cuda.is_available() and cfg['allow_cuda'] else 'cpu'
    train_loader,val_loader,test_loader=loaders(cfg); rows=[];log={}
    teacher=CNN(cfg['teacher_channels'],cfg['teacher_hidden']).to(dev)
    print('Teacher parameters:',params(teacher),flush=True)
    log['teacher']=train(teacher,train_loader,val_loader,dev,cfg['teacher_epochs'],cfg['lr'],tag='teacher')
    teacher=teacher.cpu();rows.append(record(teacher,'teacher','teacher',test_loader,cfg));write_rows(rows)
    for mode in ('global','layerwise'):
        for sparsity in cfg['sparsities']:
            pruned=copy.deepcopy(teacher);masks=unstructured_mask(pruned,sparsity,mode)
            apply_masks(pruned,masks);pruned=pruned.to(dev)
            # fine-tune at 0 also to ensure consistent protocol (and exact lab requirement)
            log[f'{mode}_{sparsity}']=train(pruned,train_loader,val_loader,dev,cfg['finetune_epochs'],cfg['finetune_lr'],masks={k:v.to(dev) for k,v in masks.items()},tag=f'{mode}_{sparsity}')
            pruned=pruned.cpu(); rows.append(record(pruned,f'{mode}_{str(sparsity).replace(".","p")}',mode,test_loader,cfg,sparsity));write_rows(rows)
    for level in cfg['structured_levels']:
        pruned=structured_channel_prune(teacher,level).to(dev)
        log[f'structured_{level}']=train(pruned,train_loader,val_loader,dev,cfg['finetune_epochs'],cfg['finetune_lr'],tag=f'structured_{level}')
        pruned=pruned.cpu();rows.append(record(pruned,f'structured_{str(level).replace(".","p")}', 'structured',test_loader,cfg,level));write_rows(rows)
    for distill in (False,True):
        name='student_distilled' if distill else 'student_plain'
        # Identical initialization ensures meaningful comparison.
        seed_everything(cfg['seed']+101)
        student=CNN(cfg['student_channels'],cfg['student_hidden']).to(dev)
        t=copy.deepcopy(teacher).to(dev) if distill else None
        log[name]=train(student,train_loader,val_loader,dev,cfg['student_epochs'],cfg['lr'],teacher=t,alpha=cfg['alpha'],T=cfg['temperature'],tag=name)
        rows.append(record(student.cpu(),name,'student',test_loader,cfg));write_rows(rows)
    plots(rows);make_report(rows,cfg,dev)
    (ROOT/'results'/'training_histories.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    print('DONE: generated results/metrics.csv, plots, checkpoints/, report.md',flush=True)

if __name__=='__main__':main()
