# SPDX-FileCopyrightText: 2026 Francisco Angulo de Lafuente
# SPDX-License-Identifier: Apache-2.0
# Research-ORCID: https://orcid.org/0009-0001-1634-7063
"""CPU reproduction of R01-L from a private chronological cache or public data.

No data are distributed. --data-dir requires NeuralBench0.4.0 and an official
kit checkout; --private-cache uses the already reconstructed private features.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
import numpy as np

def extract(data_dir,kit,out):
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(kit))
    import kit_compat
    kit_compat.apply()
    from neuralbench.data import Data
    from benchmark_utils.nb_task import _task_data_config,_make_loaders,_quiet_neuro_logs
    from torch.utils.data import DataLoader,Subset
    from features import window_features
    _quiet_neuro_logs()
    cfg=_task_data_config('eeg','_sleep_onset_stream',None,data_dir,
        {'batch_size':64,'seed':0,'num_workers':0,'pin_memory':False,'persistent_workers':False})
    nb=Data(**cfg).prepare()
    ds=_make_loaders({'train':nb['train']},'cpu',lambda y:y.reshape(()),stream_by=cfg.get('stream_by'),num_workers=0)['train'].dataset
    order=np.lexsort((ds.onset,ds.record_id))
    loader=DataLoader(Subset(ds,order.tolist()),batch_size=64,shuffle=False,num_workers=0)
    columns={k:[] for k in ['feats','y','record_id','subject','elapsed']}
    for X,y,info in loader:
        assert tuple(X.shape[1:])==(4,640)
        columns['feats'].append(window_features(X.numpy()).astype(np.float32))
        columns['y'].append(y.numpy().astype(float).reshape(-1))
        for key,meta in [('record_id','record_id'),('subject','subject_id'),('elapsed','onset')]:
            columns[key].append(np.asarray(info[meta]).reshape(-1))
    d={k:np.concatenate(v) for k,v in columns.items()}
    for key in ['record_id','subject']:d[key]=np.unique(d[key],return_inverse=True)[1].astype(np.int32)
    d['elapsed']=d['elapsed'].astype(np.float64)
    np.savez_compressed(out/'train_features_private.npz',**d)
    return d

def main():
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--private-cache',type=Path);group.add_argument('--data-dir',type=Path)
    ap.add_argument('--official-kit',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--require-reference',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    if args.private_cache:
        with np.load(args.private_cache,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    else:
        if args.official_kit is None:ap.error('--data-dir requires --official-kit')
        d=extract(args.data_dir,args.official_kit,args.out)
    assert d['feats'].shape==(61303,72) and len(np.unique(d['record_id']))==412 and len(np.unique(d['subject']))==170
    assert np.isfinite(d['feats']).all() and np.isfinite(d['y']).all() and ((d['y']>=0)&(d['y']<=600)).all()
    for r in np.unique(d['record_id']):
        m=d['record_id']==r
        assert np.array_equal(d['elapsed'][m],np.arange(m.sum())*5.)
    from models import EmaRidge
    start=time.monotonic();m=EmaRidge(alpha=10.,halflives=(2,)).fit(d);seconds=time.monotonic()-start
    w={'kind':np.str_('ema'),'ridge_mean':m.core.mean,'ridge_scale':m.core.scale,'coef':m.core.coef,
       'prior_edges':m.prior.edges,'prior_table':m.prior.table,'use_prior':np.bool_(True),'halflives':np.asarray([2.],float)}
    np.savez_compressed(args.out/'weights.npz',**w)
    source=Path(__file__).resolve().parent/'submission.py'
    shutil.copyfile(source,args.out/'submission.py')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    expected='a0afd3db3e227e7cea91f5fca08b680e23405afc92f4c1c7376394eddd1862c3'
    report={'candidate':'L','rows':61303,'records':412,'participants':170,'CPU_fit_seconds':seconds,
            'numpy':np.__version__,'GPU_used':False,'weights_sha256':sha(args.out/'weights.npz'),
            'reference_weights_byte_exact':sha(args.out/'weights.npz')==expected,
            'inference_source_byte_exact':sha(source)=='67237b4986af697bbad157b7f890e2ae6b67dc0c9074b702621969d1be1b62d1',
            'no_test_or_validation_loader_iterated':True,'no_predictive_or_official_score_computed':True}
    if args.require_reference:assert report['reference_weights_byte_exact'] and report['inference_source_byte_exact']
    (args.out/'REPRODUCTION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
