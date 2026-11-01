"""Read-only dataset audit; writes derived reports, never changes source images/splits."""
import argparse, csv, hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data_dir',type=Path,required=True); p.add_argument('--output_dir',type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    lists={s:(a.data_dir/f'{s}_list.txt').read_text(encoding='utf-8-sig').split() for s in ('train','val','test')}
    memberships=defaultdict(list)
    for s,names in lists.items():
        for n in names: memberships[n].append(s)
    report={'split_counts':{s:len(v) for s,v in lists.items()},'duplicate_list_entries':{s:[n for n,c in Counter(v).items() if c>1] for s,v in lists.items()},'split_overlap':{n:s for n,s in memberships.items() if len(set(s))>1},'errors':[]}
    files={k:{p.stem:p for p in (a.data_dir/k).iterdir() if p.is_file()} for k in ('RGB','MS','Mark')}
    report['folder_counts']={k:len(v) for k,v in files.items()}
    report['unlisted']={k:sorted(set(v)-{Path(n).stem for n in memberships}) for k,v in files.items()}
    rows=[]; colors=Counter(); shapes={k:Counter() for k in files}; hashes={k:defaultdict(list) for k in files}; thumbs=[]
    for i,(name,splits) in enumerate(memberships.items()):
        base=Path(name).stem; arrays={}; ims={}
        try:
            for k in files:
                path=files[k][base]
                with Image.open(path) as im:
                    im.load(); raw=np.array(im); shapes[k][str((im.mode,raw.shape,str(raw.dtype)))]+=1
                    ims[k]=im.convert('RGB'); arrays[k]=np.array(ims[k])
                hashes[k][hashlib.sha256(path.read_bytes()).hexdigest()].append(name)
            mask=arrays['Mark']; packed=(mask[:,:,0].astype('uint32')<<16)+(mask[:,:,1].astype('uint32')<<8)+mask[:,:,2]
            unique,counts=np.unique(packed,return_counts=True)
            for c,n in zip(unique,counts): colors[(int(c)>>16,(int(c)>>8)&255,int(c)&255)]+=int(n)
            binary=np.max(np.abs(mask.astype('int16')-np.array([240,128,128])),axis=-1)<=10
            # Match OpenCV INTER_NEAREST used by the training loader (floor mapping).
            iy=np.arange(512)*binary.shape[0]//512; ix=np.arange(512)*binary.shape[1]//512
            resized=binary[np.ix_(iy,ix)]
            coords=re.search(r'_lat_([\d.]+)_lon_([\d.]+)',name)
            rows.append({'filename':name,'split':splits[0],'foreground_ratio':float(binary.mean()),'foreground_ratio_512':float(resized.mean()),'same_dimensions':len({im.size for im in ims.values()})==1,'lat':float(coords[1]) if coords else None,'lon':float(coords[2]) if coords else None,'ms_min':int(arrays['MS'].min()),'ms_max':int(arrays['MS'].max())})
            if i%25==0:
                tile=Image.new('RGB',(768,280),'white'); draw=ImageDraw.Draw(tile)
                for j,k in enumerate(('RGB','MS','Mark')):
                    im=ims[k].resize((256,256)); tile.paste(im,(256*j,24)); draw.text((256*j+3,3),f'{k} / {splits[0]} / #{i}',fill='black')
                thumbs.append(tile)
        except Exception as e: report['errors'].append({'filename':name,'error':str(e)})
        if (i+1)%50==0: print(f'audited {i+1}/{len(memberships)}',flush=True)
    report['formats']={k:dict(v) for k,v in shapes.items()}; report['mask_colors']={str(k):v for k,v in colors.items()}
    report['exact_file_duplicates']={k:[v for v in h.values() if len(v)>1] for k,h in hashes.items()}
    report['split_stats']={}
    for s in lists:
        r=[x for x in rows if x['split']==s]; vals=[x['foreground_ratio_512'] for x in r]
        report['split_stats'][s]={'images':len(r),'foreground_fraction_512':float(np.mean(vals)),'empty_masks_512':sum(v==0 for v in vals),'dimension_mismatches':sum(not x['same_dimensions'] for x in r)}
    # Distances between image centres are a leakage warning, not proof of footprint overlap.
    spatial=[]
    for x in rows:
        if x['lat'] is None: continue
        candidates=[]
        for y in rows:
            if x['split']==y['split'] or y['lat'] is None: continue
            d=6371000*math.hypot(math.radians(x['lat']-y['lat']),math.radians(x['lon']-y['lon'])*math.cos(math.radians(x['lat'])))
            candidates.append((d,y['filename'],y['split']))
        if candidates:
            d,n,s=min(candidates); spatial.append({'filename':x['filename'],'split':x['split'],'nearest_other_split':n,'other_split':s,'distance_m':d})
    report['cross_split_nearest_centres']={'under_2m':sum(x['distance_m']<2 for x in spatial),'under_5m':sum(x['distance_m']<5 for x in spatial),'under_10m':sum(x['distance_m']<10 for x in spatial),'minimum_m':min((x['distance_m'] for x in spatial),default=None)}
    for name,data in [('samples.csv',rows),('spatial_neighbours.csv',spatial)]:
        if data:
            with (a.output_dir/name).open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (a.output_dir/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    if thumbs:
        sheet=Image.new('RGB',(768,280*len(thumbs)),'white')
        for i,t in enumerate(thumbs): sheet.paste(t,(0,i*280))
        sheet.save(a.output_dir/'alignment_samples.jpg')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
