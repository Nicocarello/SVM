import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs

x,y = make_blobs(n_samples = 50, centers = 2,random_state=0,cluster_std=0.6)

plt.scatter(x[:,0], x[:,1], c=y,s=50,cmap='autumn')

xx= np.linspace(-1,3.5)

plt.scatter(x[:,0], x[:,1], c=y,s=50,cmap='autumn')

plt.plot([0.5],[2.2],'x',color='blue')

for a,b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:

    yy= a*xx+b

    plt.plot(xx, yy, '-k')

plt.xlim(-1, 3.5)

##MAXIMIZACION DEL MARGEN

xx= np.linspace(-1,3.5)

plt.scatter(x[:,0], x[:,1], c=y,s=50,cmap='autumn')

plt.plot([0.5],[2.2],'x',color='blue')

for a,b,d in [(1,0.65,0.33),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:

    yy= a*xx+b

    plt.plot(xx, yy, '-k')

    plt.fill_between(xx, yy-d, yy+d,edgecolor='none',color='#AAAAAA',alpha=0.3)

plt.xlim(-1, 3.5)


## CREACION MODELO SVM

from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(x,y)

def plt_svc(model,ax=None,plot_support=True):
    '''PLOT DE LA FUNCION DE DECISION PARA UNA CLASIFICACION EN 2D CON SVC'''
    if ax is None:
        ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_xlim()
    
    #GENERO LOS PUNTOS PARA EVALUAR EL MODELO
    
    xx= np.linspace(xlim[0],xlim[1],30)
    yy= np.linspace(ylim[0],ylim[1],30)
    
    y,x = np.meshgrid(yy,xx)
    xy = np.vstack([x.ravel(),y.ravel()]).T
    p= model.decision_function(xy).reshape(x.shape)
    
    #REPRESENTO LAS FORNTERAS Y MARGENES DEL SVC
    
    ax.contour(xx,yy,p,colors='k',levels=[-1,0,1],alpha = 0.5,linestyles=['--','-','--'])
    
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],
                   model.support_vectors_[:,1],
                   s=300,linewidth=1,facecolors=None)
    
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
plt.scatter(x[:,0], x[:,1], c=y,s=50,cmap='autumn')
plt_svc(model,plot_support=True)


    