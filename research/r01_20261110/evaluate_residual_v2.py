"""Preregistered two-comparison, joint participant-cluster bootstrap gate."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def evaluate(rows,protocol,reps=None):
    assert protocol['gate']['individual_CI_coverage']==.975
    assert protocol['gate']['two_candidate_Bonferroni_family_error_at_most']==.05
    assert {r['regime'] for r in rows}=={'unseen','seen'}
    assert len({(r['regime'],r['record']) for r in rows})==len(rows)
    for r in rows:assert all(np.isfinite(r[k]) for k in ['H','L','N'])
    people=sorted({r['participant'] for r in rows});index={p:i for i,p in enumerate(people)}
    # Candidate x regime x participant. Sums/counts preserve equal RECORDING weight.
    sums=np.zeros((2,2,len(people)),np.float64);counts=np.zeros((2,len(people)),np.float64)
    scores={}
    for j,regime in enumerate(['unseen','seen']):
        group=[r for r in rows if r['regime']==regime]
        scores[regime]={k:float(np.mean([r[k] for r in group])) for k in ['H','L','N']}
        for r in group:
            p=index[r['participant']];counts[j,p]+=1
            for a,k in enumerate(['L','N']):sums[a,j,p]+=r['H']-r[k]
    n=protocol['gate']['bootstrap_reps'] if reps is None else reps
    rng=np.random.default_rng(protocol['gate']['bootstrap_seed'])
    draws=rng.integers(0,len(people),size=(n,len(people)))
    multiplicities=np.stack([np.bincount(d,minlength=len(people)) for d in draws]).astype(np.float64)
    denominator=multiplicities@counts.T
    assert np.all(denominator>0)
    result={}
    for a,k in enumerate(['L','N']):
        regime_gain={regime:scores[regime]['H']-scores[regime][k] for regime in ['unseen','seen']}
        gain=float(np.mean(list(regime_gain.values())))
        boot=(multiplicities@sums[a].T/denominator).mean(axis=1)
        ci=np.quantile(boot,[.0125,.9875])
        conditions={'macro_gain_at_least_0_5s':bool(gain>=.5),'no_regime_regression':bool(min(regime_gain.values())>=0),
                    'Bonferroni_97_5CI_lower_strictly_positive':bool(ci[0]>0)}
        result[k]={'macro_gain_seconds':gain,'regime_gain_seconds':regime_gain,'paired_participant_cluster_CI_97_5':ci.tolist(),
                   'conditions':conditions,'eligible':all(conditions.values())}
    eligible=[k for k in ['L','N'] if result[k]['eligible']]
    selected=max(eligible,key=lambda k:(result[k]['macro_gain_seconds'],k=='L')) if eligible else None
    return {'scores_by_regime':scores,'macro_scores':{k:float(np.mean([scores[g][k] for g in scores])) for k in ['H','L','N']},
            'candidates':result,'selected':selected,'participants':len(people),'paired_record_count':len(rows),'bootstrap_reps':n,
            'bootstrap_seed':protocol['gate']['bootstrap_seed'],'two_candidates_family_error_control':'Bonferroni individual97.5% CIs; family<=5%',
            'no_sealed_or_public_test_or_validation_metric':True,'only_repaired_public_training_CV':True,'official_score_unknown':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',required=True);p.add_argument('--protocol',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    rows=Path(a.rows);protocol=Path(a.protocol);out=Path(a.out)
    result=evaluate(json.loads(rows.read_text()),json.loads(protocol.read_text()))
    result.update(paired_private_rows_sha256=hashlib.sha256(rows.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    with out.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
