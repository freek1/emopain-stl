"""Fresh STAL-Stacked reproduction with explicit paper and archival profiles.

The default paper profile uses paper hyperparameters and recovered original
SNN window normalization, sampling, loss and recurrent-LIF operations. The
archival profile retains the earlier cleaned-repo full-recording comparison.
Neither profile claims regenerated published aggregate scores.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

import importlib.metadata
import platform
from utils.paper_pipeline import paper_config
from train import main
from utils.load_data import load_data_emopain


def write_json(path, obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    temp.replace(path)


def evaluate(out):
    streams=[]
    densities=[]
    for modality in ['emg','energy','angle']:
        path=out/'results'/'fresh'/f'results_{modality}_STL-S.csv'
        if not path.exists():return
        df=pd.read_csv(path).set_index('fold').sort_index()
        if len(df)!=46 or df.index.nunique()!=46:return
        streams.append(df)
        densities.append(float(df['sparsity'].mean()))
    labels=[df['test_labels'].map(lambda s:int(s.strip('[]'))).to_numpy() for df in streams]
    assert all(np.array_equal(labels[0],y) for y in labels[1:])
    X=np.column_stack([df['test_preds'].map(lambda s:int(s.strip('[]'))).to_numpy() for df in streams])
    y=labels[0]
    np.random.seed(42)
    model=StackingClassifier(estimators=[(m,RandomForestClassifier()) for m in ['emg','energy','angle']],final_estimator=RandomForestClassifier())
    model.fit(X,y)
    pred=model.predict(X)
    result=dict(scope='Fresh 46-recording LOO base predictions; historical in-sample ensemble scoring',accuracy=float(accuracy_score(y,pred)),auc_hard_labels=float(roc_auc_score(y,pred)),f1_binary=float(f1_score(y,pred,zero_division=0)),f1_macro=float(f1_score(y,pred,average='macro',zero_division=0)),mcc=float(matthews_corrcoef(y,pred)),confusion_matrix=confusion_matrix(y,pred).tolist(),density_by_modality=dict(zip(['emg','energy','angle'],densities)),harmonic_density=3/sum(1/d for d in densities),paper_reference=dict(accuracy=37/46,auc_hard_labels=.678921568627451,f1_binary=.5263157894736842,mcc=.43746312472211024,harmonic_density=.549))
    write_json(out/'metrics.json',result)
    print('FINAL_METRICS',json.dumps(result),flush=True)


def prepare_run(out, provenance, resume=False):
    """Never reuse an output directory after data, code or settings change."""
    path=out/'provenance.json'
    if path.exists():
        if not resume:
            raise ValueError('Output already exists; use --resume or a new directory.')
        if json.loads(path.read_text()) != provenance:
            raise ValueError('Resume rejected: configuration, source, data or environment differ.')
    else:
        if resume or any(out.iterdir()):
            raise ValueError('Output lacks valid provenance; use a new empty directory.')
        write_json(path,provenance)


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--modalities',nargs='+',choices=['emg','energy','angle'],default=['emg','energy','angle'])
    folds=parser.add_mutually_exclusive_group()
    folds.add_argument('--folds',nargs='+',type=int,default=None)
    folds.add_argument('--all-folds',action='store_true')
    parser.add_argument('--profile',choices=['paper','archived'],default='paper')
    parser.add_argument('--num-steps',type=int,default=None)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.threads < 1 or (args.num_steps is not None and args.num_steps < 1):
        parser.error('threads and num-steps must be positive')
    if args.num_steps is None:args.num_steps=5 if args.profile=='paper' else 10
    if args.folds is None:args.folds=list(range(46)) if args.all_folds else [0,18]
    if not args.folds or len(set(args.folds))!=len(args.folds) or any(f<0 or f>=46 for f in args.folds):
        parser.error('Choose unique folds from 0 through 45')
    if len(set(args.modalities))!=len(args.modalities):parser.error('Modalities must be unique')
    configs={}
    for modality in args.modalities:
        config=paper_config(modality)
        config['num_steps']=args.num_steps
        if args.profile=='archived':
            config.update(profile='archived', full_recording=True, learn_beta=False,
                          classifier_layout='full_recording', normalization='recording',
                          encoder_loss='weighted',sampling='recording_count',
                          checkpoint_selection='legacy_classifier_state_alias')
        configs[modality]=config
    data_dir=args.data_dir.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).parent
    source_files=[source/'train.py',*(source/'utils').glob('*.py'),Path(__file__)]
    provenance=dict(data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [data_dir/'data_Cs.pickle',data_dir/'data_Ps.pickle']},source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)},torch=torch.__version__,numpy=np.__version__,profile=args.profile,configurations=configs,python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['torch','snntorch','numpy','scikit-learn','pandas','matplotlib']},num_steps=args.num_steps,modalities=args.modalities,folds=args.folds,threads=args.threads,seed=1957,device='cpu')
    prepare_run(out,provenance,resume=args.resume)
    torch.set_num_threads(args.threads)
    torch.manual_seed(1957);np.random.seed(1957)
    checkpoint=out/'progress.pt'
    if args.resume and not checkpoint.exists():
        # No completed fold: restart from the initial RNG state.
        print('No completed-fold checkpoint; restarting the first selected fold.',flush=True)
    completed=[]
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=False)
        completed=state['completed'];torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng'])
    os.chdir(out)
    started=time.monotonic()
    total=len(args.modalities)*len(args.folds)
    try:
        for modality in args.modalities:
            X,y=load_data_emopain(modality,data_dir=data_dir,task='pain')
            if X.shape[:2]!=(46,17995) or torch.bincount(y).tolist()!=[34,12]:
                raise ValueError('Prepared data do not match the recovered 46-recording/34-negative/12-positive contract. See DATA_CONTRACT.md.')
            write_json(out/f'data_contract_{modality}.json',dict(shape=list(X.shape),labels=y.tolist(),
                       normalization='per recording per channel min-max, then profile window normalization',
                       tensor_sha256=hashlib.sha256(X.contiguous().numpy().tobytes()).hexdigest()))
            for fold in args.folds:
                key=f'{modality}:{fold}'
                if key in completed:continue
                for path in [out/'results'/'fresh'/'spiketrains',out/'imgs'/'fresh']:path.mkdir(parents=True,exist_ok=True)
                result_path=out/'results'/'fresh'/f'results_{modality}_STL-S.csv'
                if not result_path.exists():
                    pd.DataFrame(columns=['fold','train_acc','val_acc','test_acc','test_preds','test_labels','sparsity']).set_index('fold').to_csv(result_path)
                else:
                    frame=pd.read_csv(result_path)
                    frame=frame[frame['fold'].map(lambda f:f'{modality}:{int(f)}' in completed)]
                    frame.to_csv(result_path,index=False)
                config=configs[modality]
                write_json(out/f'config_{modality}.json',config)
                print('FOLD_START',key,'completed',len(completed),'of',total,flush=True)
                tick=time.monotonic()
                main(config,X,y,fold,np.array([i for i in range(46) if i!=fold]),np.array([fold]),torch.device('cpu'),'fresh')
                row=pd.read_csv(result_path).iloc[-1].to_dict()
                completed.append(key)
                state=dict(completed=completed,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state())
                temp=checkpoint.with_suffix('.tmp');torch.save(state,temp);temp.replace(checkpoint)
                write_json(out/'status.json',dict(status='running',completed=len(completed),total=total,last_fold=key,last_result=row,last_fold_seconds=time.monotonic()-tick,elapsed_this_process_seconds=time.monotonic()-started))
                print('FOLD_COMPLETE',key,'seconds',time.monotonic()-tick,'result',json.dumps(row),flush=True)
                # Encoded arrays are disposable; predictions, densities and RNG state
                # are durable. No cached predictions are imported into a fresh run.
                for p in (out/'results'/'fresh'/'spiketrains').glob(f'*_{modality}_{fold}_STL-S.npy'):p.unlink()
                # Keep selected-model checkpoints and histories for auditability.
                gc.collect()
            del X,y;gc.collect()
        evaluate(out)
        write_json(out/'status.json',dict(status='complete',completed=len(completed),total=total,elapsed_this_process_seconds=time.monotonic()-started))
    except Exception:
        write_json(out/'status.json',dict(status='failed',completed=len(completed),total=total,error=traceback.format_exc()))
        raise


if __name__=='__main__':run()
