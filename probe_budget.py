"""Compare fresh 20/50-epoch schedules using train/validation ONLY on Colab.

No test dataset is instantiated. Each budget starts from the same seed.
Interrupted runs restart that run; completed runs are reused only for identical config.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time


def select_budget(runs, budgets, seeds, min_gain):
    """Predeclared conservative rule: longer must improve mean and every seed."""
    means = {b: statistics.mean(runs[(b,s)]['best_val_IoU_GD'] for s in seeds) for b in budgets}
    selected = min(budgets)
    decisions=[]
    for b in sorted(budgets):
        if b == selected: continue
        gains=[runs[(b,s)]['best_val_IoU_GD']-runs[(selected,s)]['best_val_IoU_GD'] for s in seeds]
        accepted=statistics.mean(gains)>=min_gain and all(g>0 for g in gains)
        decisions.append({'short_budget':selected,'long_budget':b,'paired_gain_pp':gains,'mean_gain_pp':statistics.mean(gains),'accepted':accepted})
        if accepted: selected=b
    return selected, means, decisions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir',required=True,type=Path)
    p.add_argument('--output_dir',required=True,type=Path)
    p.add_argument('--cache_dir',default='/content/cache_budget_probe',type=Path)
    p.add_argument('--budgets',type=int,nargs='+',default=[20,50])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,123])
    p.add_argument('--num_workers',type=int,default=2)
    p.add_argument('--min_gain_pp',type=float,default=0.5)
    args=p.parse_args()
    if len(set(args.seeds))<2 or len(set(args.budgets))<2 or min(args.budgets)<1:
        p.error('Need >=2 unique seeds and >=2 positive budgets.')
    args.seeds=sorted(set(args.seeds)); args.budgets=sorted(set(args.budgets))
    import torch
    from torch.utils.data import DataLoader
    from rice_data.dataset import GDRGBMSDataset, resolve_data_dir
    from models.unet_model import build_model
    from utils.engine import set_seed, seed_worker, build_criterion, _make_scaler, evaluate
    if not torch.cuda.is_available():
        raise SystemExit('Enable a GPU runtime in Colab before running this probe.')
    args.data_dir=Path(resolve_data_dir(str(args.data_dir)))
    split_hash=hashlib.sha256(b''.join((args.data_dir/f'{s}_list.txt').read_bytes() for s in ('train','val'))).hexdigest()
    config={'mode':'ms_only','arch':'unet','backbone':'resnet34','encoder_weights':'imagenet','img_size':512,'batch_size':4,'lr':1e-4,'weight_decay':1e-4,'loss':'dice_bce','augment':True,'threshold':0.5,'scheduler':'CosineAnnealingLR','eta_min':1e-6,'split_sha256':split_hash,'selection_metric':'val_IoU_GD','test_used':False,'deterministic':True}
    config['versions']={n:importlib.metadata.version(n) for n in ('torch','torchvision','segmentation-models-pytorch','numpy','opencv-python')}
    # Record source provenance so changed loaders/models cannot silently reuse results.
    here=Path(__file__).parent
    config['source_sha256']={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('probe_budget.py','configs/config.py','rice_data/dataset.py','models/unet_model.py','utils/engine.py','utils/metrics.py')}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config_path=args.output_dir/'probe_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=config:
        raise SystemExit('Output contains a different configuration. Use a NEW output_dir.')
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    cache=args.cache_dir/split_hash[:12]
    datasets=[GDRGBMSDataset(str(args.data_dir),f'{s}_list.txt','ms_only',(512,512),augment=s=='train',cache_dir=str(cache),preload=True,verbose=True) for s in ('train','val')]
    device=torch.device('cuda'); runs={}
    for budget in args.budgets:
        for seed in args.seeds:
            out=args.output_dir/f'epochs_{budget}'/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
            result_path=out/'validation_result.json'
            if result_path.exists():
                r=json.loads(result_path.read_text(encoding='utf-8'))
                if r['config']!=config or r['budget']!=budget or r['seed']!=seed:
                    raise RuntimeError(f'Configuration mismatch: {result_path}')
                runs[(budget,seed)]=r; print(f'Reuse {result_path}',flush=True); continue
            set_seed(seed,deterministic=True)
            generator=torch.Generator().manual_seed(seed)
            kw=dict(batch_size=4,num_workers=args.num_workers,pin_memory=True,worker_init_fn=seed_worker,persistent_workers=args.num_workers>0)
            train=DataLoader(datasets[0],shuffle=True,generator=generator,**kw)
            val=DataLoader(datasets[1],shuffle=False,**kw)
            model=build_model('unet','resnet34',3,1,encoder_weights='imagenet').to(device)
            criterion=build_criterion('dice_bce')
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=budget,eta_min=1e-6)
            scaler=_make_scaler(True); best=-1.; best_epoch=0; start=time.time()
            with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','val_loss','val_IoU_GD','val_F1_GD','lr']); writer.writeheader()
                for epoch in range(1,budget+1):
                    model.train(); total=0.; n=0
                    for images,masks in train:
                        images=images.to(device,non_blocking=True); masks=masks.to(device,non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type='cuda',enabled=True): logits=model(images)
                        loss=criterion(logits.float(),masks)
                        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                        total+=loss.item()*len(images); n+=len(images)
                    metrics=evaluate(model,val,device,criterion,use_amp=False)
                    row={'epoch':epoch,'train_loss':total/n,'val_loss':metrics['loss'],'val_IoU_GD':metrics['IoU_GD'],'val_F1_GD':metrics['F1_GD'],'lr':optimizer.param_groups[0]['lr']}
                    if not all(__import__('math').isfinite(row[k]) for k in ('train_loss','val_loss','val_IoU_GD')):
                        raise RuntimeError('Non-finite training/validation value; run not accepted.')
                    writer.writerow(row); f.flush(); scheduler.step()
                    if metrics['IoU_GD']>best:
                        best=metrics['IoU_GD']; best_epoch=epoch
                        torch.save(model.state_dict(),out/'best_model.pth')
                    print(f'budget={budget} seed={seed} epoch={epoch} val_IoU_GD={metrics["IoU_GD"]:.3f} best={best:.3f}',flush=True)
            r={'config':config,'budget':budget,'seed':seed,'best_epoch':best_epoch,'best_val_IoU_GD':best,'minutes':(time.time()-start)/60,'device':torch.cuda.get_device_name(0)}
            temp=result_path.with_suffix('.tmp'); temp.write_text(json.dumps(r,indent=2),encoding='utf-8'); os.replace(temp,result_path)
            runs[(budget,seed)]=r
            del model,optimizer,scaler,train,val; torch.cuda.empty_cache()
    selected,means,decisions=select_budget(runs,args.budgets,args.seeds,args.min_gain_pp)
    summary={'status':'validation_probe_complete','selected_budget':selected,'mean_best_val_IoU_GD':means,'decisions':decisions,'rule':f'Prefer shorter; accept longer only if mean gain >= {args.min_gain_pp} percentage points and positive gain on EVERY paired seed. Heuristic, not a significance test.','seeds':args.seeds,'common_config':config,'early_stopping':False,'note':'Use selected fixed budget for all modalities. Keep checkpoint with highest validation IoU_GD. Never select using test.'}
    (args.output_dir/'selected_protocol.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
