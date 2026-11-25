import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image
plt.rcParams.update({"font.family":["Liberation Sans","DejaVu Sans"]})
W,H=1340,1380
fig=plt.figure(figsize=(W/300,H/300),dpi=300); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off')
c=np.array([670,690]); R=400
def pill(xy,txt,fc,tc,bold=False):
    w=len(txt)*30+80
    ax.add_patch(FancyBboxPatch((xy[0]-w/2,xy[1]-36),w,72,boxstyle="round,pad=0,rounding_size=36",fc=fc,ec='none',zorder=3))
    ax.text(*xy,txt,ha='center',va='center',fontsize=12.5,color=tc,weight='bold' if bold else 'normal',zorder=4)
# trans axes: Thr35 (up) - O2B (down); Ser17 (upper right) - O2G (lower left); W (upper left) - W (lower right)
lig=[("Thr35 OG1 / Water",90,'#FCE4E4','#B00000','#D00000','-', '2.12'),
     ("GTP O2B",270,'#E3EFF9','#1F4E79','0.35','-','1.85'),
     ("Ser17 OG",30,'#EBEBEB','0.15','0.35','-','2.08'),
     ("GTP O2G",210,'#E3EFF9','#1F4E79','0.35','-','1.78'),
     ("Water",150,'#EBEBEB','0.15','0.35',(0,(2,1.5)),None),
     ("Water",330,'#EBEBEB','0.15','0.35',(0,(2,1.5)),None)]
for name,ang,fc,tc,lc,ls,dist in lig:
    a=np.deg2rad(ang); u=np.array([np.cos(a),-np.sin(a)])
    p=c+u*R; p0=c+u*120
    ax.plot([p0[0],p[0]-u[0]*42],[p0[1],p[1]-u[1]*42],color=lc,lw=2.4 if 'Thr' in name else 1.8,ls=ls,zorder=1)
    pill(p,name,fc,tc,bold='Thr' in name)
    if dist:
        q=c+u*235+np.array([-u[1],u[0]])*58
        ax.text(*q,dist,ha='center',va='center',fontsize=12,color='#C00000' if 'Thr' in name else '0.35',zorder=4)
ax.add_patch(Circle(c,120,fc='#EBA4A4',ec='#A85555',lw=1.5,zorder=5))
ax.text(*c,'Mg²⁺',ha='center',va='center',fontsize=14,weight='bold',color='0.15',zorder=6)
ax.text(670,215,'water when Switch I opens',ha='center',va='center',fontsize=12,color='0.35')
ax.text(670,1270,'Mg²⁺ first shell · reference MD means, Mg–O (Å)',ha='center',va='center',fontsize=12.5,color='0.1')
ax.text(670,1340,'opposite pairs: Thr35–O2B, Ser17–O2G, water–water',ha='center',va='center',fontsize=9.5,color='0.4')
fig.savefig('fig1b.png',dpi=300); print(Image.open('fig1b.png').size)
