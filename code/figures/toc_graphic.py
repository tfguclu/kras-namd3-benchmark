import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":["Liberation Sans","DejaVu Sans"]})
z=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', 'kras_ca', 'toc_frames.npz'))
nca=int(z['nca']); ng=int(z['ngtp']); res=z['res']
C,O=z['closed'],z['open']
def parts(X): return X[:nca],X[nca:nca+ng],X[nca+ng],X[nca+ng+1:]
core=~(((res>=30)&(res<=40))|((res>=60)&(res<=76)))&(res>=1)&(res<=166)
def kabsch(P,Q):
    pc,qc=P.mean(0),Q.mean(0); H=(P-pc).T@(Q-qc); U,S,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(Vt.T@U.T)); R=Vt.T@np.diag([1,1,d])@U.T; return R,pc,qc
R,pc,qc=kabsch(O[:nca][core],C[:nca][core]); O=(O-pc)@R.T+qc
caC,gC,mgC,tC=parts(C); caO,gO,mgO,tO=parts(O)
cen=caC.mean(0)
s1=(res>=30)&(res<=40)
x=mgC-cen; x/=np.linalg.norm(x)
dv=caO[s1].mean(0)-caC[s1].mean(0); dv-= (dv@x)*x; y=dv/np.linalg.norm(dv)
zz=np.cross(x,y)
def proj(P): Q=(P-cen); return np.c_[Q@y, Q@x, Q@zz]    # screen x = switch displacement, screen y = toward Mg
def smooth(P,n=8):
    out=[]
    for i in range(len(P)-1):
        p0=P[max(i-1,0)];p1=P[i];p2=P[i+1];p3=P[min(i+2,len(P)-1)]
        for t in np.linspace(0,1,n,endpoint=False):
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t**3))
    out.append(P[-1]); return np.array(out)
fig=plt.figure(figsize=(2,2)); ax=fig.add_axes([0,0,1,1]); ax.axis('off'); ax.set_aspect('equal')
pC=proj(caC); pO=proj(caO)
# protein body (closed frame) excluding Switch I, depth cued
m=(res>=1)&(res<=166)
idx=np.where(m)[0]
S=smooth(pC[idx]); rr=np.repeat(res[idx],8)[:len(S)]
zmin,zmax=S[:,2].min(),S[:,2].max()
for i in range(len(S)-1):
    if 27<=rr[i]<=41: continue
    f=(S[i,2]-zmin)/(zmax-zmin)
    col=str(0.55+0.3*(1-f)) ; lw=2.2+1.2*f
    ax.plot(S[i:i+2,0],S[i:i+2,1],color=col,lw=lw,solid_capstyle='round',zorder=2+f)
def loop(P,col,ls,lw,zo,al=1):
    s=(res>=26)&(res<=42)
    Q=smooth(P[s]); ax.plot(Q[:,0],Q[:,1],color=col,lw=lw,ls=ls,solid_capstyle='round',zorder=zo,alpha=al)
loop(pC,'#0868AC','-',2.6,6,0.9)
loop(pO,'#D55E00','-',3.0,7)
# GTP and Mg
g=proj(gC); gn=list(z['gnames'])
GC='#E08A00'
mgp=proj(mgC[None])[0][:2]
base=mgp+np.array([-1.5,-9.5])
mg=proj(mgC[None])[0]; ax.add_patch(Circle(mg[:2],2.1,color='#2E8B57',zorder=10))
ax.text(mg[0],mg[1],'Mg²⁺',fontsize=4.3,color='white',weight='bold',ha='center',va='center',zorder=11)
t=proj(tC)[1]; to=proj(tO)[1]
ax.add_patch(Circle(t[:2],0.9,fc='#0868AC',ec='white',lw=0.4,zorder=9))
ax.add_patch(Circle(to[:2],0.9,fc='#D55E00',ec='white',lw=0.4,zorder=9))
ax.plot([mg[0],t[0]],[mg[1],t[1]],color='#0868AC',lw=0.8,ls=(0,(1.5,1)),zorder=9)
# frame
allp=np.r_[pC[m][:,:2],pO[m][:,:2]]
lo,hi=allp.min(0)-3,allp.max(0)+3
span=max(hi-lo); c=(lo+hi)/2
top=span*0.36
ax.set_xlim(c[0]-span/2-1,c[0]+span/2+1); ax.set_ylim(c[1]-span/2-span*0.2,c[1]+span/2+top)
X0,X1=ax.get_xlim(); Y0,Y1=ax.get_ylim(); W=X1-X0; H=Y1-Y0
ax.add_patch(FancyBboxPatch((X0+0.03*W,Y1-0.17*H),0.94*W,0.14*H,boxstyle=f"round,pad=0,rounding_size={0.04*W}",fc='#0868AC',ec='none',zorder=20))
ax.text(X0+W/2,Y1-0.10*H,'3.4× faster KRAS MD',color='white',fontsize=9,weight='bold',ha='center',va='center',zorder=21)
lc=smooth(pC[(res>=26)&(res<=42)]); lo_=smooth(pO[(res>=26)&(res<=42)])
ax.text(lc[:,0].mean()-1, lc[:,1].max()+2.0,'closed',color='#0868AC',fontsize=5.5,weight='bold',ha='center',zorder=21)
ax.text(lo_[:,0].mean()+1, lo_[:,1].max()+2.0,'open',color='#D55E00',fontsize=5.5,weight='bold',ha='center',zorder=21)
ax.text(X0+W/2,Y0+0.115*H,'Mg²⁺ site intact',color='#2E8B57',fontsize=7,weight='bold',ha='center',zorder=21)
ax.text(X0+W/2,Y0+0.035*H,'Switch I rearranges',color='#D55E00',fontsize=7,weight='bold',ha='center',zorder=21)
for e in ('png','tiff'): fig.savefig(f'TOC_graphic.{e}',dpi=600,**({'pil_kwargs':{'compression':'tiff_lzw'}} if e=='tiff' else {}))
fig.savefig('TOC_graphic.pdf'); print(mg, t, to)
