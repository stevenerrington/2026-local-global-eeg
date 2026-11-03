"""
20_frontal_generator_check.py — is the left insula / inferior-frontal part of the local-effect cluster a
second MMN generator, or spread from auditory cortex?

Part 1: signed eLORETA (LD − LS) time courses, sign-flip-aligned label means, for left/right auditory
        cortex (HG + STG), insula + pars opercularis, and parahippocampal + inferior temporal; window
        t-tests (80–140 ms) and jackknife peak-latency difference (insula/IFG − auditory cortex).
Part 2: leakage simulation — a coherent source confined to left auditory cortex is projected through
        the forward model and inverted with each participant's inverse operator; the insula/IFG label
        amplitude relative to auditory cortex is the share expected from spread alone.
Run from the project root (needs the same inputs as 19_figure3.py). Results are printed.
"""
import sys; sys.argv=["x"]
from pathlib import Path
ROOT_TMP = Path("derivatives/source/frontal_check_tcs.npz")
import importlib.util, numpy as np, mne
from scipy import stats
spec=importlib.util.spec_from_file_location("m","scripts/19_figure3.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
mne.set_log_level("ERROR")
subs=sorted(f.stem for f in m.CACHE_DIR.glob("sub-*.npz"))
info=mne.io.read_info(m.CACHE_DIR/"info.fif"); info=mne.pick_info(info,mne.pick_types(info,eeg=True))
with info._unlock(): info["projs"]=[]
raw0=mne.io.RawArray(np.zeros((len(info["ch_names"]),1)),info); raw0.set_eeg_reference("average",projection=True); info=raw0.info
fwd=m.make_forward(info); src=fwd["src"]
labs={lb.name:lb for lb in mne.read_labels_from_annot(m.TEMPLATE,"aparc",subjects_dir=m.SUBJECTS_DIR)}
R={"AC":["transversetemporal","superiortemporal"],"IFG_ins":["insula","parsopercularis"],"ITG_PHG":["parahippocampal","inferiortemporal"]}
out={}
for sub in subs:
    z=np.load(m.CACHE_DIR/f"{sub}.npz"); cov=mne.read_cov(m.SRC_DIR/f"{sub}-cov.fif")
    inv=mne.minimum_norm.make_inverse_operator(info,fwd,cov,loose=m.LOOSE,depth=m.DEPTH)
    ev=mne.EvokedArray((z["topo__LD"]-z["topo__LS"])*1e-6,info,tmin=z["times"][0])
    stc=mne.minimum_norm.apply_inverse(ev,inv,m.LAMBDA2,method="eLORETA",pick_ori="normal",method_params=dict(max_iter=60))
    for r,names in R.items():
        for h in ("lh","rh"):
            lb=sum((labs[f"{n}-{h}"] for n in names[1:]),labs[f"{names[0]}-{h}"])
            out.setdefault((r,h),[]).append(stc.extract_label_time_course(lb,src,mode="mean_flip")[0])
t=stc.times
np.savez(str(ROOT_TMP),t=t,**{f"{r}_{h}":np.stack(v) for (r,h),v in out.items()})
w=(t>=0.08)&(t<=0.14)
for k,v in out.items():
    X=np.stack(v); s=np.sign(X.mean(0)[w].mean()); X=X*s
    tt=stats.ttest_1samp(X[:,w].mean(1),0)
    print(k, f"t={tt.statistic:.2f} p={tt.pvalue:.4f}", "peak", round(t[(t>.05)&(t<.2)][np.argmax(X.mean(0)[(t>.05)&(t<.2)])]*1e3), "corr w AC", np.round(np.corrcoef(np.abs(np.stack(out[('AC',k[1])]).mean(0)), np.abs(X.mean(0)))[0,1],3))

# ---------------- part 2
mne.set_log_level("ERROR")
z=np.load(ROOT_TMP); t=z["t"]; w=(t>=0.06)&(t<=0.16)
def sa(X): return X*np.sign(X.mean(0)[(t>=.08)&(t<=.14)].mean())
AC, IF = sa(z["AC_lh"]), sa(z["IFG_ins_lh"])
ratio_obs = IF.mean(0)[w].max()/AC.mean(0)[w].max()
# jackknife peak-latency difference (IFG − AC)
n=len(AC); tw=t[w]
def pk(X): return tw[np.argmax(X.mean(0)[w])]
d_full=pk(IF)-pk(AC)
jk=np.array([pk(np.delete(IF,i,0))-pk(np.delete(AC,i,0)) for i in range(n)])
se=np.sqrt((n-1)/n*((jk-jk.mean())**2).sum()); tj=d_full/se if se>0 else np.nan
print(f"obs IFG/AC peak ratio {ratio_obs:.2f}; peak lat AC {pk(AC)*1e3:.0f} IFG {pk(IF)*1e3:.0f}; diff {d_full*1e3:.0f} ms, jackknife SE {se*1e3:.1f} ms, t={tj:.2f}, p={2*stats.t.sf(abs(tj),n-1):.3f}")
# leakage simulation: unit dipoles in left AC only -> forward -> eLORETA -> label means
subs=sorted(f.stem for f in m.CACHE_DIR.glob("sub-*.npz"))
info=mne.io.read_info(m.CACHE_DIR/"info.fif"); info=mne.pick_info(info,mne.pick_types(info,eeg=True))
with info._unlock(): info["projs"]=[]
raw0=mne.io.RawArray(np.zeros((len(info["ch_names"]),1)),info); raw0.set_eeg_reference("average",projection=True); info=raw0.info
fwd=m.make_forward(info); src=fwd["src"]
fwd_fixed=mne.convert_forward_solution(fwd,surf_ori=True,force_fixed=True,use_cps=True)
labs={lb.name:lb for lb in mne.read_labels_from_annot(m.TEMPLATE,"aparc",subjects_dir=m.SUBJECTS_DIR)}
ac=labs["transversetemporal-lh"]+labs["superiortemporal-lh"]; ifg=labs["insula-lh"]+labs["parsopercularis-lh"]
rat=[]
for sub in subs[:15]:
    cov=mne.read_cov(m.SRC_DIR/f"{sub}-cov.fif")
    inv=mne.minimum_norm.make_inverse_operator(info,fwd,cov,loose=m.LOOSE,depth=m.DEPTH)
    # source amplitude pattern: equal-sign normal currents across AC label (as mean_flip assumes coherent source)
    vert=[s["vertno"] for s in src]; x=np.zeros(sum(len(v) for v in vert))
    idx=np.where(np.isin(vert[0],ac.vertices))[0]; x[idx]=1e-9
    # flip so that the simulated source matches the mean_flip convention of the label
    flip=mne.label_sign_flip(ac,src); x[idx]*=flip
    data=fwd_fixed["sol"]["data"]@x
    ev=mne.EvokedArray(np.tile(data[:,None],(1,3)),info,tmin=0)
    ev.set_eeg_reference("average",projection=True)
    stc=mne.minimum_norm.apply_inverse(ev,inv,m.LAMBDA2,method="eLORETA",pick_ori="normal",method_params=dict(max_iter=60))
    a=abs(stc.extract_label_time_course(ac,src,mode="mean_flip")[0][0]); b=abs(stc.extract_label_time_course(ifg,src,mode="mean_flip")[0][0])
    rat.append(b/a)
print("leakage AC->IFG/insula ratio: mean %.2f, range %.2f-%.2f"%(np.mean(rat),min(rat),max(rat)))
