import numpy as np
import matplotlib.pyplot as plt
import pylab as py
import InverseSamp as samp
import imp;
imp.reload(samp)

plt.ion()
plt.close()
params = {'legend.fontsize': 12}
py.rcParams.update(params)
fig, axs = plt.subplots(2,1,figsize=(6, 8), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0, bottom=0.1, wspace=0,left=0.13,right=0.95,top=0.99)
axs=axs.ravel()


## READ IN THE TEST DATA ##
xx,yy,zz = np.loadtxt('testdata.txt',usecols=(0,1,2), unpack='true')

#set data distribution that we want to sample from
data=zz
min=np.min(data)-1
max=np.max(data)+1

#set the binning
sbins = np.arange(min,max,0.2)

#number of Monte Carlo (MC) samples
Nsamp=500
#Number of points to sample in each MC
Nran=len(data)


mclab='%d MC samples'%(Nsamp)
MCtot=np.zeros([len(sbins)-1])
MCbins=np.zeros([Nsamp,len(sbins)-1])
for i in range(Nsamp):
    ss = samp.InverseSamp(data,Nran,80)
    #plots each MC histogram (red)
    axs[0].hist(ss,sbins,histtype='step',color='r',alpha=0.5)
    #if statement used purely for printing legend properly
    if i==Nsamp-2:
        axs[0].hist(ss,sbins,histtype='step',color='r',alpha=0.5,label=mclab)
    #keep track of bin information to compute residuals later on
    n,mcbinloc  = np.histogram(ss,sbins)
    MCtot +=n
    MCbins[i,:] = n

#bin the input data
ndatabin, bins, = np.histogram(data,sbins)

#compute normalised residual between data and MC sampled data
#mean of MC samples
MCmean = MCtot/float(Nsamp)
#variance
VarMC = 1./(Nsamp-1.) * np.sum((MCmean - MCbins)**2,axis=0)
top =MCmean - ndatabin
bottom=np.sqrt(VarMC)
R=top/bottom
#remove inf caused by 0/0
R[np.isnan(R)] = 0.
R[R<-10e10] = 0.

#plot data - black line
axs[0].hist(data,sbins,histtype='step',color='k',linewidth=2,label='data')
#plot mean of MC samples
meanbin = (sbins[:-1]+sbins[1:])/2
axs[0].plot(meanbin,MCmean,'y',linewidth=1.5,label='mean MC')

#plot residuals
mcbinloc = (mcbinloc[:-1]+mcbinloc[1:])/2
axs[1].plot(mcbinloc,R,'k')
axs[1].axhline(y=0,linestyle='--')
axs[1].axhline(y=-3,linestyle=':')
axs[1].axhline(y=3,linestyle=':')


maxy=np.max(n)+100
axs[0].set_xlim([min,max])
axs[0].set_ylim([0,maxy])
axs[0].set_ylabel('N')
axs[0].minorticks_on()
axs[0].set_xticklabels(())
axs[1].minorticks_on()
axs[1].set_xlim([min,max])
axs[1].set_ylim([-4,4])
axs[1].set_xlabel('data')
axs[1].set_ylabel('normalised residual')


lg = axs[0].legend(loc='upper left',ncol=1,fontsize=10)
lg.draw_frame(False)

plt.show()

plt.savefig('example.png',dpi=300)
