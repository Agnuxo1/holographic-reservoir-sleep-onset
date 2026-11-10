"""Frozen paired H/L/N study; pilot fits training only and emits no utility score."""
from pathlib import Path
import argparse,hashlib,json,os,resource,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from models import EmaRidge,metric_weights,wbmae

root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','full'],required=True)
mode=ap.parse_args().mode
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
torch.set_num_threads(4);torch.set_num_interop_threads(1)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
def verify_source():
    assert {p.name for p in root.iterdir() if p.is_file()}==set(manifest)|{'SOURCE_MANIFEST.json'}
    for name,digest in manifest.items():assert sha(root/name)==digest,name
verify_source();protocol=json.loads((root/'PROTOCOL.json').read_text())
assert protocol['candidate_N']['epochs']==80 and protocol['gate']['individual_CI_coverage']==.975
def read(name):
    with np.load(root/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}
data=read('train_features_private.npz');sampled=read('sampled_control_private.npz')
assert len(data['y'])==len(sampled['y'])==61303
out=Path('/mnt/datos/proyectos/eeg-sealed-20261110/residual-study-v2')/mode
out.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
(out/'STARTED.json').write_text(json.dumps({'mode':mode,'manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'protocol':protocol,'torch':torch.__version__},indent=2)+'\n')

def folds(regime):
    rng=np.random.default_rng(protocol['fold_seed'])
    if regime=='unseen':
        subjects=np.unique(data['subject']);rng.shuffle(subjects)
        return [np.isin(data['subject'],subjects[f::5]) for f in range(5)]
    pool=[]
    for s in np.unique(data['subject']):
        records=np.unique(data['record_id'][data['subject']==s])
        if len(records)>=2:pool.append(records[rng.integers(len(records))])
    pool=np.array(pool);rng.shuffle(pool)
    return [np.isin(data['record_id'],pool[f::5]) for f in range(5)]
def subset(d,mask):return {k:v[mask] for k,v in d.items()}
def design(model,d):
    Z=model._inputs(d['feats'],d['elapsed'],d['record_id']);baseline=model.core.predict(Z)
    Z=np.concatenate([np.clip((Z-model.core.mean)/model.core.scale,-10,10),baseline[:,None]/600.0],axis=1)
    assert Z.shape==(len(baseline),151) and np.isfinite(Z).all()
    return Z.astype(np.float32),baseline.astype(np.float32)
class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(151,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,base):return torch.clamp(base+80*torch.tanh(self.net(x).squeeze(-1)),0,600)

rows=[];max_vram=0;training_times=[];fit_times=[]
for regime in (['unseen'] if mode=='pilot' else protocol['regimes']):
    for fold,held in enumerate(folds(regime)):
        if mode=='pilot' and fold!=0:continue
        tr=subset(data,~held);train_records=np.unique(tr['record_id'])
        assert set(train_records).isdisjoint(set(data['record_id'][held]))
        if regime=='unseen':assert set(tr['subject']).isdisjoint(set(data['subject'][held]))
        else:assert set(data['subject'][held]).issubset(set(tr['subject']))
        fit_started=time.monotonic();linear=EmaRidge(alpha=10.0,halflives=(2,)).fit(tr)
        fit_times.append({'kind':'L','seconds':time.monotonic()-fit_started})
        if mode=='full':
            sh=subset(sampled,np.isin(sampled['record_id'],train_records))
            fit_started=time.monotonic();historical=EmaRidge(alpha=10.0,halflives=(2,)).fit(sh)
            fit_times.append({'kind':'H','seconds':time.monotonic()-fit_started})
        Z,b=design(linear,tr);w=metric_weights(tr['y'],tr['record_id']).astype(np.float32)
        assert np.isclose(w.sum(),1,rtol=0,atol=1e-6)
        Xt=torch.from_numpy(Z).cuda();bt=torch.from_numpy(b).cuda();yt=torch.from_numpy(tr['y'].astype(np.float32)).cuda();wt=torch.from_numpy(w).cuda()
        predictions=[]
        for seed in ([11] if mode=='pilot' else protocol['seeds']):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=Residual().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            torch.cuda.reset_peak_memory_stats();train_start=time.monotonic()
            epochs=5 if mode=='pilot' else 80
            for epoch in range(epochs):
                model.train();indices=torch.randperm(len(Xt),device='cuda')
                for idx in indices.split(4096):
                    optimizer.zero_grad(set_to_none=True)
                    pred=model(Xt[idx],bt[idx]);loss=(wt[idx]*len(Xt)*torch.abs(pred-yt[idx])).mean()
                    assert torch.isfinite(loss)
                    loss.backward();optimizer.step()
            torch.cuda.synchronize();elapsed=time.monotonic()-train_start
            training_times.append(elapsed);max_vram=max(max_vram,torch.cuda.max_memory_reserved())
            if mode=='full':
                te=subset(data,held);Zv,bv=design(linear,te)
                model.eval()
                with torch.inference_mode():vp=model(torch.from_numpy(Zv).cuda(),torch.from_numpy(bv).cuda()).cpu().numpy()
                assert np.isfinite(vp).all();predictions.append(vp)
                torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'regime':regime,'fold':fold,'seed':seed,'protocol_sha256':sha(root/'PROTOCOL.json')},out/f'{regime}-f{fold}-s{seed}.pt')
            del optimizer,model
            print(json.dumps({'mode':mode,'regime':regime,'fold':fold,'seed':seed,'epochs':epochs,'train_seconds':elapsed,'peak_reserved_bytes':max_vram,'pilot_has_no_heldout_scoring':mode=='pilot'}),flush=True)
        if mode=='full':
            nonlinear=np.mean(predictions,axis=0)
            h=historical.predict_feats(te['feats'],te['elapsed'],te['record_id'])
            for record in np.unique(te['record_id']):
                m=te['record_id']==record
                rows.append({'regime':regime,'fold':fold,'record':int(record),'participant':int(te['subject'][m][0]),'n_windows':int(m.sum()),
                    'H':wbmae(te['y'][m],h[m]),'L':wbmae(te['y'][m],bv[m]),'N':wbmae(te['y'][m],nonlinear[m])})
            (out/'PAIRED_RECORDS_PRIVATE.json').write_text(json.dumps(rows,indent=2)+'\n')
        del Xt,bt,yt,wt
        torch.cuda.empty_cache()
verify_source()
result={'mode':mode,'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'seconds':time.monotonic()-started,
 'GPU_training_seconds':training_times,'CPU_IRLS_fit_times':fit_times,
 'max_reserved_VRAM_GiB':max_vram/1024**3,'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
 'GPU_used':True,'device':torch.cuda.get_device_name(0),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
 'CPU_affinity_count':len(os.sched_getaffinity(0)),'no_public_test_or_validation_split_access':True,
 'no_pilot_heldout_predictions_or_scores':mode=='pilot','paired_records':len(rows),'gate_evaluated':False}
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
