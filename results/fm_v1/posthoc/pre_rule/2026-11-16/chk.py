import csv, io, json, re, sys, zipfile
from collections import defaultdict, Counter
import numpy as np
C=["CSC","MH","MacTel","Stargardt","RP"]
z=zipfile.ZipFile(sys.argv[1]); names=z.namelist()
print(Counter(re.sub(r"fold\d_k\d+_draw\d+","EP",n).rsplit("/",1)[0] for n in names if "/external/" in n and "MIRAGE" in n))
pat=re.compile(r"runs/([^/]+)/(development|reporting)/(prototype|logistic)/fold(\d)_k(\d+)_draw(\d+)/(predictions\.csv|episode\.json|fit\.json)$")
ep={}; fits=Counter(); rows=defaultdict(list); cnt=Counter()
for n in names:
    m=pat.match(n)
    if not m: continue
    arm,role,clf,f,k,d,fn=m.groups(); f,k,d=int(f),int(k),int(d)
    if fn=="episode.json":
        j=json.loads(z.read(n)); ep[(arm,role,clf,f,k,d)]=(json.dumps(j["supports"]),json.dumps(j["queries"]))
    elif fn=="fit.json":
        j=json.loads(z.read(n)); fits[(j.get("converged"),j.get("state"))]+=1
    else:
        cnt[(arm,role,clf,k)]+=1
        for r in csv.DictReader(io.TextIOWrapper(z.open(n),encoding="utf-8")):
            rows[(arm,role,clf,k,d)].append((f,r["query_id"],r["true_label"],r["predicted_label"]))
print("fits",fits)
print("counts",sorted(Counter((r,c,k,v) for (a,r,c,k),v in cnt.items()).items()))
# pairing
bad=0;n=0
for key,v in ep.items():
    arm,role,clf,f,k,d=key
    if clf!="logistic": continue
    n+=1
    if ep.get((arm,role,"prototype",f,k,d))!=v: bad+=1
    if ep.get(("MIRAGE",role,"logistic",f,k,d))!=v: bad+=1
print("logistic episodes",n,"mismatch vs prototype/MIRAGE",bad)
def macro(rs):
    by=defaultdict(list)
    for f,q,t,p in rs: by[t].append(t==p)
    return np.mean([np.mean(by[c]) for c in C])
def cls(rs):
    by=defaultdict(list)
    for f,q,t,p in rs: by[t].append(t==p)
    return np.array([np.mean(by[c]) for c in C])
print("arm role K | n25 diff se t | wins/ties/losses | sdP25 sdL25 sdP100 | corr | perclass diff")
allcells=[]
for arm in ["MIRAGE","DINOv2","Inception","RETFound"]:
  for role,ks in [("development",[4]),("reporting",[1,2,4,8,16])]:
    for k in ks:
        p=np.array([macro(rows[(arm,role,"prototype",k,d)]) for d in range(25)])
        l=np.array([macro(rows[(arm,role,"logistic",k,d)]) for d in range(25)])
        p100=np.array([macro(rows[(arm,role,"prototype",k,d)]) for d in range(100)])
        df=l-p; se=df.std(ddof=1)/5
        pc=np.mean([cls(rows[(arm,role,"logistic",k,d)])-cls(rows[(arm,role,"prototype",k,d)]) for d in range(25)],0)
        nq=len(rows[(arm,role,"prototype",k,0)])
        print(f"{arm:9s} {role[:3]} K={k:2d} nq={nq} | {df.mean():+.3f} se {se:.3f} t={df.mean()/se if se else 0:+.1f} | {int((df>1e-9).sum())}/{int((abs(df)<1e-9).sum())}/{int((df<-1e-9).sum())} | {p.std(ddof=1):.3f} {l.std(ddof=1):.3f} {p100.std(ddof=1):.3f} p100mean {p100.mean():.3f} p25mean {p.mean():.3f}| r={np.corrcoef(p,l)[0,1]:.2f} | "+" ".join(f"{x:+.2f}" for x in pc))
        allcells.append((arm,role,k,df))
# MIRAGE-minus-others under logistic vs prototype (does ranking change?)
for role,k in [("development",4),("reporting",4),("reporting",8),("reporting",16)]:
    out=[]
    for clf in ["prototype","logistic"]:
        ms={a:np.mean([macro(rows[(a,role,clf,k,d)]) for d in range(25)]) for a in ["MIRAGE","DINOv2","Inception","RETFound"]}
        out.append(" ".join(f"{a}={v:.3f}" for a,v in ms.items()))
    print(role,k,"proto:",out[0],"| logit:",out[1])
# external five-way
pe=re.compile(r"runs/([^/]+)/external/(OCTID|OCTDL)/(prototype|logistic)/fold(\d)_k(\d+)_draw(\d+)/predictions\.csv$")
er=defaultdict(list)
for n in names:
    m=pe.match(n)
    if not m: continue
    arm,src,clf,f,k,d=m.groups(); d=int(d)
    if d>=25: continue
    for r in csv.DictReader(io.TextIOWrapper(z.open(n),encoding="utf-8")):
        er[(arm,src,clf,int(k),d)].append((r["true_label"],r["predicted_label"]))
def stat(rs):
    by=defaultdict(list)
    for t,p in rs: by[t].append(t==p)
    return {c:np.mean(v) for c,v in by.items()}, len(rs)
print("EXTERNAL five-way: recall on mapped true labels, logistic vs prototype, 25 draws")
for arm in ["MIRAGE","DINOv2","Inception","RETFound"]:
  for src in ["OCTID","OCTDL"]:
    for k in [1,4,8,16]:
        if (arm,src,"prototype",k,0) not in er: continue
        res={}
        for clf in ["prototype","logistic"]:
            acc=defaultdict(list)
            for d in range(25):
                s,nq=stat(er[(arm,src,clf,k,d)])
                for c,v in s.items(): acc[c].append(v)
            res[clf]={c:np.mean(v) for c,v in acc.items()}
        print(arm,src,"K",k,"n",nq," ".join(f"{c}: P{res['prototype'][c]:.3f} L{res['logistic'][c]:.3f}" for c in res["prototype"]))
