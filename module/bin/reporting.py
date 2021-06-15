"""
.. module:: reporting
   :synopsis: creates reporting.
.. moduleauthor:: Sebastian Multsch <smultsch@knoell.com>
"""

# native
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime,timedelta
from shutil import copyfile,make_archive
import xml.etree.ElementTree as ET
from collections import namedtuple

# data and stats
import pandas as pd
import numpy as np
import h5py

# plotting
import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import collections  as mcc
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
from mpl_toolkits.mplot3d import Axes3D


###############################################################################
# functions data i/o

def get_fdir(subfolder):
    """ Return current working directoy and add subfolder to path. """
    fdir = os.path.abspath(
                os.path.join(
                    os.path.dirname(Path(__file__).parent),
                    *subfolder))
    return fdir

def parse_info(fp):
    """ converts attributes-xml into python object"""
    tree = ET.parse(fp)
    root = tree.getroot()
    attr = dict([(child.tag, child.text) for child in root])
    # convert to Python if needed
    attr["t0"] = datetime.strptime(attr["t0"],"%Y-%m-%dT%H:%M")
    attr["tn"] = datetime.strptime(attr["tn"],"%Y-%m-%dT%H:%M")
    attr["reaches"] = attr["reaches"].split(",")
    attr["pec_p1"] = eval(attr["pec_p1"])
    attr["pec_p2"] = eval(attr["pec_p2"])
    attr["pec_func"] = eval(attr["pec_func"])
    attr["pec_ylim"] = eval(attr["pec_ylim"])
    attr["pec_ylim_small"] = eval(attr["pec_ylim_small"])    
    attr["lguts_p1"] = eval(attr["lguts_p1"])
    attr["lguts_p2"] = eval(attr["lguts_p2"])
    attr["lguts_func"] = eval(attr["lguts_func"])
    attr["lguts_ylim"] = eval(attr["lguts_ylim"])
    Attributes = namedtuple('Attributes', sorted(attr))
    attr = Attributes(**attr)
    return attr

###############################################################################
# functions statistics
    
def cumHist(Z):
    """Get cumulative distribution function"""
    values, X2 = np.histogram(Z, bins=10)
    F2 = np.cumsum(values)
    return X2[:-1],F2
     
def get_rolling_window(a, window):
    """
    Get a rolling window for a certain period.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def calc_CDF_Weibull(N):
    """
    Calculates CDF by Weilbull: 1/(N+1)
    """
    return np.array(range(1,N+1))/float(N+1)

def calc_interpolated_probability(F,Z,x):
    """
    Calculates a value related to a specific probability of a Weibull CDF
    by linear interpolation.
    F=Probabilities CDF
    Z=Sorted values.
    x=Target probability
    """
    return np.interp(x, F, Z)

def calc_Probabilities_Weibull(Z,prob):
    """
    """
    Z.sort()
    F = calc_CDF_Weibull(Z.shape[0])    
    return calc_interpolated_probability(F,Z,prob)

def get_values_vulnerability(res,perc1,perc2,func):    
    """ Get values for vulnerability plot """
    # get values
    maxvals = np.array([np.sort(func(res[i,:,:],axis=1)) for i in range(res.shape[0])])
    median = np.median(maxvals,axis=0)
    perc80 = np.percentile(maxvals,perc1,axis=0)
    perc20 = np.percentile(maxvals,perc2,axis=0)
    return median,perc80,perc20

def get_annual(data,year,hourly, daily,freq="hourly"):
    """ Return annual data of cmfcont """
    if freq == "hourly":
        maxind = data.shape[2]
        t_year = [i for i in [t if val.year==year else -9999 for t,val 
                              in enumerate(hourly)] if i !=-9999 and i<maxind]
    elif freq == "daily":
        t_year = [i for i in [t if val.year==year else -9999 for t,val 
                  in enumerate(daily)] if i !=-9999]
    return  data[:,:,t_year]  

def get_twa(data,window):
    """ calculate TWA values""" 
    if type(data) == np.ndarray:
        return np.array([[np.mean(get_rolling_window(data[mc,r,:],window=window),axis=1)  for r in range(data.shape[1])] for mc in range(data.shape[0])])
    else:
        return None
###############################################################################
# class + functions data

class Data:
    def __init__(self,fpath,key,cmfcont,steps,cascade,lguts,t0,tn):
        """ reads data directly from aqRisk@LandscapeModel projects or from
        pre-prepared HDF5-File."""
        
        #set variables
        self.fpath = fpath
        self.t0 = t0
        self.tn = tn
        
        # get folder of Monte-Carlo simulations
        if os.path.isfile(os.path.join(self.fpath,"SpraydriftList.csv")):
            self.mcs = list(pd.read_csv(os.path.join(self.fpath,"SpraydriftList.csv"),usecols=["mc"])["mc"].unique())
        else:
            self.mcs = os.listdir(os.path.join(self.fpath,"mcs"))
        
        # check if HDF5 exists, other read from source and creeate HDF5
        if not os.path.isfile(os.path.join(self.fpath,"res.h5")):
            # get mcs's
            self.mcs = os.listdir(os.path.join(self.fpath,"mcs"))
            # create hdf5
            with h5py.File(os.path.join(self.fpath,"res.h5"), 'w') as hdf:
                # reach CMF
                if cmfcont == 'true':
                    
                    # read cmf
                    component = "cmfcont"
                    param="pecsw"
                    name = "e1"
                    data = np.array([self.__read_cmfcont(os.path.join(self.fpath,"mcs",mc,"processing", "efate",  component),name,mc) for mc in self.mcs])
                    hdf.create_dataset(component+"_"+param,data=data,compression="gzip",compression_opts=4)   
                    data = np.array([self.__read_cmf(os.path.join(self.fpath,"mcs",mc,"processing", "efate",  component),name,mc) for mc in self.mcs])
                    hdf.create_dataset("cmf"+"_"+"depth",data=data,compression="gzip",compression_opts=4)   
  
                    # read lguts
                    if lguts == 'true':
                        param = "survival"
                        data = np.array([self.__read_lguts(os.path.join(self.fpath,"mcs",mc,"processing", "effect", "lguts_"+component,"c"),mc) for mc in self.mcs])
                        hdf.create_dataset(component+"_"+param,data=data,compression="gzip",compression_opts=4)    
                        
                        # read reachnames from lguts
                        fpath_lguts = os.path.join(self.fpath,"mcs",self.mcs[0],"processing",
                                                   "effect", "lguts_"+component,"c")
                        reaches = np.loadtxt(os.path.join(fpath_lguts,"0.csv"), delimiter=',', usecols=[0],dtype=str)    
                        np.savetxt(os.path.join(self.fpath,"reaches_lguts.csv"),reaches,fmt='%10s',delimiter=",")
                    
                    # read spraydrift
                    data = pd.concat([self.__read_spraydrift(os.path.join(self.fpath,"mcs",mc,"processing", "efate",  component),name,mc) for mc in self.mcs])
                    data.to_csv(os.path.join(self.fpath,"SpraydriftList.csv"),sep=",")
                    
                    # copy reach list
                    reachlist = pd.read_csv(os.path.join(self.fpath,"mcs",self.mcs[0],"processing","efate", component,name,"ReachList.csv"),)[["key","x","y","downstream"]]
                    reachlist.sort_values(by="key",inplace=True)
                    reachlist.set_index("key",inplace=True)
                    reachlist.to_csv(os.path.join(self.fpath,"ReachList.csv"),sep=",")
                    
                    # copy catchmetn list       
                    catchmentlist = pd.read_csv(os.path.join(self.fpath,"mcs",self.mcs[0],"processing","efate",component,name,"CatchmentList.csv")   ,index_col="key")
                    catchmentlist.to_csv(os.path.join(self.fpath,"CatchmentList.csv"),sep=",")
                    
                # read steps
                if steps == 'true':
                    component = "steps"
                    name = "h1"
                    param = "pecsw"
                    data = np.array([self.__read_steps(os.path.join(self.fpath,"mcs",mc,"processing", "efate",  component),name,mc) for mc in self.mcs])
                    hdf.create_dataset(component+"_"+param,data=data,compression="gzip",compression_opts=4)    
                    # read lguts
                    if lguts == 'true':
                        param = "survival"
                        data = np.array([self.__read_lguts(os.path.join(self.fpath,"mcs",mc,"processing", "effect", "lguts_"+component,"c"),mc) for mc in self.mcs])
                        hdf.create_dataset(component+"_"+param,data=data,compression="gzip",compression_opts=4)    
                        
                        # read reachnames from lguts
                        fpath_lguts = os.path.join(self.fpath,"mcs",self.mcs[0],"processing",
                                                   "effect", "lguts_"+component,"c")
                        reaches = np.loadtxt(os.path.join(fpath_lguts,"0.csv"), delimiter=',', usecols=[0],dtype=str)    
                        np.savetxt(os.path.join(self.fpath,"reaches_lguts.csv"),reaches,fmt='%10s',delimiter=",")  
       
                         # copy reach list
                        reachlist = pd.read_csv(os.path.join(self.fpath,"mcs",self.mcs[0],"processing","efate", component,name,"ReachList.csv"),)[["key","x","y","downstream"]]
                        reachlist.sort_values(by="key",inplace=True)
                        reachlist.set_index("key",inplace=True)
                        reachlist.to_csv(os.path.join(self.fpath,"ReachList.csv"),sep=",")
                        
                        # copy catchmetn list       
                        catchmentlist = pd.read_csv(os.path.join(self.fpath,"mcs",self.mcs[0],"processing","efate",component,name,"CatchmentList.csv")   ,index_col="key")
                        catchmentlist.to_csv(os.path.join(self.fpath,"CatchmentList.csv"),sep=",")       
        
        
                if cascade == "true":
                    component = "cascade"
                    name = "e1"
                    param = "pecsw"
                    data = np.array([self.__read_cascade(os.path.join(self.fpath,"mcs",mc,"processing", "efate",  component,"experiments",name),mc) for mc in self.mcs])
                    hdf.create_dataset(component+"_"+param,data=data,compression="gzip",compression_opts=4)    
                    self.cascade = data
                    # read lguts
                    if lguts == 'true':
                        param = "survival"
                        data = np.array([self.__read_lguts(os.path.join(self.fpath,"mcs",mc,"processing", "effect", "lguts_"+component,"c"),mc) for mc in self.mcs])
                        hdf.create_dataset(component+"_"+param,data=data,compression="gzip",compression_opts=4)    
                        
                        # read reachnames from lguts
                        fpath_lguts = os.path.join(self.fpath,"mcs",self.mcs[0],"processing",
                                                   "effect", "lguts_"+component,"c")
                        reaches = np.loadtxt(os.path.join(fpath_lguts,"0.csv"), delimiter=',', usecols=[0],dtype=str)    
                        np.savetxt(os.path.join(self.fpath,"reaches_lguts.csv"),reaches,fmt='%10s',delimiter=",")  

        # read data
        try:
            # open HDF5
            fp=os.path.join(self.fpath,"res.h5")
            print("read",fp)
            self.res = h5py.File(fp, "r")

            # read other
            self.reachlist = pd.read_csv(os.path.join(self.fpath,"ReachList.csv"))
            self.catchmentlist = pd.read_csv(os.path.join(self.fpath,"CatchmentList.csv"))
            self.spraydriftlist = pd.read_csv(os.path.join(self.fpath,"SpraydriftList.csv"))
            self.spraydriftlist.time = pd.to_datetime(self.spraydriftlist.time,format="%Y-%m-%dT%H:%M")
            self.lguts_reaches = np.loadtxt(os.path.join(self.fpath,"reaches_lguts.csv"), delimiter=',', usecols=[0],dtype=str) 
            self.lguts_reaches = [i.strip() for i in self.lguts_reaches]
        except:
            print("Unexpected error:", sys.exc_info()[0])
            
        # create time arrays
        self.hourly=pd.date_range( datetime(self.t0.year,self.t0.month,self.t0.day),
                             datetime(self.tn.year,self.tn.month,self.tn.day)+timedelta(23/24),freq="H")
        self.daily=pd.date_range( datetime(self.t0.year,self.t0.month,self.t0.day),
                            datetime(self.tn.year,self.tn.month,self.tn.day))       
        self.year_ind = [(i,day) for i,day in enumerate(self.daily)]
        self.year_ind = [[i[0] for i in self.year_ind if i[1].year==yr] for yr in self.daily.year.unique()]   
        self.year_ind = [(min(i),max(i)+1) for i in self.year_ind]
        

        # create downstream connections
        outlet = self.catchmentlist.iloc[0]   
        coords = []
        for index, reach in self.reachlist.iterrows():
            if reach.downstream == "Outlet":
                x2 = outlet.x
                y2= outlet.y
            else:
                down = self.reachlist[self.reachlist["key"]==reach["downstream"]]
                x2 = down.x.values[0]
                y2 = down.y.values[0]
            coords.append([x2,y2])
        coords = np.array(coords)
        self.reachlist["downstream_x"] =coords[:,0]
        self.reachlist["downstream_y"] =coords[:,1]

    def __read_cmf(self,fpath,fname,mc):
        """Reads results of cmfcont"""
        print("read","hydro",mc)
        data = pd.read_csv(os.path.join(fpath,fname,fname+"_reaches.csv"),
                           usecols=["key","time"]+["depth"])
        data ["time"]=pd.to_datetime(data ["time"],format="%Y-%m-%dT%H:%M")
        data.set_index(["key","time"],inplace=True)
        data_sorted = data.sort_index(level=(0,1))
        data = data_sorted.values.reshape(data.index.levels[0].shape[0],
                                   data.index.levels[1].shape[0])
        return data
      
    def __read_spraydrift(self,fpath,fname,mc):
        """ read spraydrift """
        spraydrift = pd.read_csv(os.path.join(fpath,fname,"SpraydriftList.csv"))
        spraydrift["time"]=pd.to_datetime(spraydrift["time"],format="%Y-%m-%dT%H:%M")
        spraydrift["mc"] = mc
        spraydrift.sort_values(by="key",inplace=True)
        spraydrift.set_index("mc",inplace=True)
        return spraydrift
        
    def __read_cmfcont(self,fpath,fname,mc):
        """Reads results of cmfcont"""
        print("read","cmfcont",mc)
        data = pd.read_csv(os.path.join(fpath,fname,fname+"_reaches.csv"),
                           usecols=["key","time"]+["PEC_SW"])
        data ["time"]=pd.to_datetime(data ["time"],format="%Y-%m-%dT%H:%M")
        data.set_index(["key","time"],inplace=True)
        data_sorted = data.sort_index(level=(0,1))
        data = data_sorted.values.reshape(data.index.levels[0].shape[0],
                                   data.index.levels[1].shape[0])
        return data

    def __read_lguts(self,fpath,mc):
        """ Reads results of lguts """
        print("read","lguts",mc)
        files = [f for f in os.listdir(fpath) if f.split(".")[1]=="txt"]
        data = np.array([np.loadtxt(os.path.join(fpath,f),delimiter='\t',dtype=float) for f in files])
        data = np.concatenate([data[i] for i in  range(data.shape[0])]).T
        return data
    
    def __read_steps(self,fpath,fname,mc):
        """ read steps results"""
        print("read","steps",mc)
        fp=os.path.join(fpath,fname,fname+"_reaches.h5")
        return h5py.File(fp, "r") ["PEC_SW"][:].T

    def __read_cascade(self,fpath,mc):
        """ reads results for mcascade """
        def read(fpath,fp):
            fp = files[0]
            f = open(os.path.join(fpath,fp))
            f = f.read()
            f = f.split("\n")[1:-2]
            return [float(i.split(",")[2]) for i in f]
        print("read","cascade",mc)
        files = os.listdir(fpath)
        files = [i for i in files if len(i.split("."))>1 and i.split(".")[1]=="csv"]
        files = [i for i in files if i.find("diagnostics")<0] 
        files = sorted(files)
        res = np.array([read(fpath,fp) for fp in files])
        return res



###############################################################################
# functions single plots
    
def plot_reach_timeseries(fout,key,mc,component,reach,t0,tn,
                          daily,hourly,days,depth,pec,drift,survival,
                          survival_ylim=(0,1.1)):
    """ Plots depth,pec,drift and survival over time for one reach"""

  
    # create figure
    fig = plt.figure(figsize=[12,6])
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    offset=0
        
    #plot depth
    par = host.twinx()
    new_fixed_axis = par.get_grid_helper().new_fixed_axis
    par.axis["right"] = new_fixed_axis(loc="right", axes=par,offset=(offset, 0))
    par.plot(hourly,depth[:min(len(depth),len(hourly))], label="depth", color="k",linestyle="-",alpha=.5,linewidth=0.5)
    par.set_xlim(t0,tn)
    offset+=60
    par.set_ylabel("Depth [m]")
    
    #plot PEC
    par = host.twinx()
    new_fixed_axis = par.get_grid_helper().new_fixed_axis
    par.axis["right"] = new_fixed_axis(loc="right", axes=par,offset=(offset, 0))
    par.plot(hourly,pec[:min(len(pec),len(hourly))], label="pec",color="r",linestyle="-",linewidth=2,alpha=1)
    offset+=60
    par.set_ylabel("PEC$_{SW}$ [$\mu$g L$^{-1}$]")

    #plot spraydrift
    par = host.twinx()
    new_fixed_axis = par.get_grid_helper().new_fixed_axis
    par.axis["right"] = new_fixed_axis(loc="right", axes=par,offset=(offset, 0))
    par.plot(drift.time.values,drift.rate.values,
             label="Drift",color="orange",linewidth=0,marker="D",alpha=1,markersize=10)
    offset+=60
    par.set_ylabel("Drift [mg m$^{-2}$]")

    # get splits to plot each year separately
    def get_splits(vals):
        splits=[0]
        i_split=0
        for i,val in enumerate(vals[:-1]):

            diff = vals[i]-vals[i+1]
            if diff<0:
                i_split+=1
            splits.append(i_split)
        vals_ind = np.arange(vals.shape[0])
        splits = np.array([vals_ind[splits==i] for i in np.unique(splits)]) 
        return splits
    splits = get_splits(survival)
    

    xt = daily

    
    if min(survival) == 1:
        host.plot(xt,survival,color="g",linewidth=2,alpha=1)
        
    else:
    
        for x,y in zip( [xt[i] for i in splits],[survival[i] for i in splits]):
            p1, = host.plot(x,y,color="g",linewidth=2,alpha=1)
            


            
    host.set_xlabel("Time")
    host.set_ylabel("Survival [0:dead; 1:alive]")   
    host.set_ylim(survival_ylim)
    host.set_title(key+"_"+mc+"_"+component+"_"+reach)

    # create legend    
    leg2 = mlines.Line2D([],[],color='k', label="Depth",alpha=1,linewidth=1,linestyle="--")
    leg4 = mlines.Line2D([],[],color='r', label="PEC$_{SW}$",alpha=1,linewidth=3,linestyle="-")
    leg5 = mlines.Line2D([],[],color='g', label="Survival",alpha=1,linewidth=3,linestyle="-")
    leg6 = mlines.Line2D([],[],color='orange', label="Drift",linewidth=0,marker="D",alpha=1)
    plt.legend(handles=[leg2,leg4,leg5,leg6],ncol=5, bbox_to_anchor=(0.00, 1.05, .45, .102),
               frameon=True)
    
    #save figure  
    plt.close("all")
    fig.savefig(os.path.join(fout,key+"_"+mc+"_"+component+"_"+reach+".png"),dpi=300)        
             
def plot_ConcFreqDur(fout,key,mc,component,dat,threshold=0.00001,
                     pec_y=""):
    """ makes a duration-concentration-frequency plot"""
    
    print("plot_ConcFreqDur",mc,component)
    a=dat.flatten()
    a[a<threshold]=0
    idx = np.where(a!=0)[0]
    aout = np.split(a[idx],np.where(np.diff(idx)!=1)[0]+1)
    x = [len(i) for i in aout]
    y = [np.median(i) for i in aout]
    # get histogram
    H, xedges, yedges = np.histogram2d(x, y,normed=True,bins=20)
    # make plot
    fig = plt.figure(figsize=(14,4))
    xx,yy = np.meshgrid(xedges[:-1], yedges[:-1])
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xx, yy, H,shade=True,alpha=.75)#,color=attr.pec_c,alpha=.5)
    ax.set_xlabel("Duration [hours]")
    ax.set_ylabel(pec_y)
    ax.set_zlabel("Frequency [normed]")    
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(xx[:8,:], yy[:8,:], H[:8,:],shade=True,alpha=.75)#,color=attr.pec_c,alpha=.5)
    ax.set_xlabel("Duration [hours]")
    ax.set_ylabel(pec_y)
    ax.set_zlabel("Frequency [normed]")    
    plt.tight_layout()
    plt.close("all")
    fig.savefig(os.path.join(fout,key+"_"+mc+"_"+component+".png"),dpi=300) 
    
def plot_vulnerability(median,ylim,perc80,perc20,xlabel,ylabel,color,label,
                       year,p1,p2,key,component,fpath,
                       twa_median=None,twa_perc80=None,twa_perc20=None):
    """ Make a vulnerability plot."""

    # make figure
    fig, ax = plt.subplots(1, 1, sharex=True,figsize=[6,6])
    # plot median and percentiles
    ax.plot(median,color=color)    
    ax.fill_between(range(0,median.shape[0]),perc20,perc80, 
                     facecolor=color, 
         edgecolors="None",alpha=.25,label="20-80% percentile") 
    if type(twa_median) == np.ndarray:
        # plot median and percentiles
        ax.plot(twa_median,color="darkblue")    
        ax.fill_between(range(0,median.shape[0]),twa_perc20,twa_perc80, 
                         facecolor="darkblue", 
             edgecolors="None",alpha=.25,label="20-80% percentile")     
    ax.grid(True,alpha=.2,color="k")    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True,alpha=.2,color="k")  
    ax.set_xticks(range(0,median.shape[0],int(median.shape[0]/5)))
    ax.set_ylim(ylim)

    ax.text(.05,.95,key+"_"+component+"_"+str(year),
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,fontweight="bold")
    # create legend   
    if not type(twa_median) == np.ndarray:
        leg1= mlines.Line2D([],[],color=color, label=label+": median",alpha=1,
                            linewidth=3,linestyle="-")
        leg2=mpatches.Patch(color=color,
                            label=label+": %ith-%ith perc."%(p1,p2),alpha=.5)
        leg=plt.legend(handles=[leg1,leg2],ncol=2, 
                       bbox_to_anchor=(0.5, 1.05, .45, .102),
                   frameon=True)    
    else:
        leg1= mlines.Line2D([],[],color=color, label=label+": median",alpha=1,
                            linewidth=3,linestyle="-")
        leg2=mpatches.Patch(color=color,
                            label=label+": %ith-%ith perc."%(p1,p2),alpha=.5)
        leg3= mlines.Line2D([],[],color="darkblue", label="PEC$_{SW,twa}$"+": median",alpha=1,
                            linewidth=3,linestyle="-")
        leg4=mpatches.Patch(color="darkblue",
                            label="PEC$_{SW,twa}$"+": %ith-%ith perc."%(p1,p2),alpha=.5)
        leg=plt.legend(handles=[leg1,leg2,leg3,leg4],ncol=2, 
                       bbox_to_anchor=(0.6, 1.04, .45, .102),
                   frameon=True)    
    leg._legend_box.align = "left"
    plt.close("all")
    fig.savefig(fpath,dpi=300)   

def map_vulnerability(fout,key,mc,component,year,surf,xup,yup,xdown,ydown,res):
    """ Make map of vulnerability """

    
    # make figure
    fig= plt.figure(figsize=(6,5))
    ax1 = fig.add_axes([0.15,0.1,0.75,0.75]) # x,y, lenght, height    
    
    #create a colorbar
    cmap = matplotlib.cm.get_cmap('jet_r')
    norm = matplotlib.colors.Normalize(vmin=attr.lguts_ylim[0],
                               vmax=attr.lguts_ylim[1])
    colors = [cmap(norm(val)) for val in surf]
    
    # create map
    ax1.set_xlim((xup.min()),xup.max())
    ax1.set_ylim((yup.min(),yup.max()))
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.grid(True)    
    
    # set tile
    ax1.text(0.5,0.95,str(year),
            horizontalalignment='center',verticalalignment="top",
            transform=ax1.transAxes,fontweight="bold")
   
    # add data
    lines = [[x,y] for x,y in zip(res[["x","y"]].values,
                         res[["downstream_x","downstream_y"]].values)]
            
            
            
    lc = mcc.LineCollection(lines, colors=colors, linewidths=2)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.margins(0.1)
    ax1.set_title(key+"_"+mc+"_"+component+"_"+str(year))
    # make colormap
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cb=fig.colorbar(sm,cax=cax, orientation='vertical')
    cb.set_label("Survival [-]")
    plt.close("all")
    fig.savefig( os.path.join(fout,key+"_"+mc+"_"+component+"_lguts_map_"+str(year)+".png"),dpi=300,transparent=True) 
    fig.clf()    

###############################################################################
# function to create all plots of model run
    
def make_plots(attr,fout,mcs,component,t0,tn,hourly, daily, pecsw,pecsw_twa,survival,
               depth, reachlist,lguts_reaches, spraydriftlist):
    """ Make several plots ...."""



    
    # plot results for each year
    for y,year in enumerate(range(t0.year,tn.year+1,1)):
 
        # plot vulnerability of PECmax and PECtwa
        print("plot_vulnerability",year,component)
        r = get_annual(pecsw,year,hourly, daily,freq="hourly") 
        median,perc80,perc20 = get_values_vulnerability(r,attr.pec_p1,
                                                            attr.pec_p2,
                                                            attr.pec_func)
        
        r = get_annual(pecsw_twa,year,hourly, daily,freq="hourly")
        twa_median,twa_perc80,twa_perc20  = get_values_vulnerability(r,
                                      attr.pec_p1,
                                      attr.pec_p2,
                                      attr.pec_func)
        
        
        fp = os.path.join(fout,attr.key+"_"+component+"_"+str(year)+".png")
        plot_vulnerability(median,attr.pec_ylim,perc80,perc20,attr.pec_x,attr.pec_y,
                          attr.pec_c,attr.pec_l,year,attr.pec_p1,
                          attr.pec_p2,attr.key,component,fp,
                          twa_median,twa_perc80,twa_perc20)
    
        if attr.lguts == "true":
            # plot vulnerability lguts
            res_lguts = get_annual(survival,year,hourly, daily,freq="daily")
            median,perc80,perc20 = get_values_vulnerability(res_lguts,attr.lguts_p1,
                                                                            attr.lguts_p2,
                                                                            attr.lguts_func)
            fp = os.path.join(fout,attr.key+"_"+component+"_lguts_"+str(year)+".png")
            print("plot_vulnerability",year,component,"lguts")
            plot_vulnerability(median,attr.lguts_ylim,perc80,perc20,attr.lguts_x,attr.lguts_y,
                              attr.lguts_c,attr.lguts_l,year,attr.pec_p1,
                              attr.pec_p2,attr.key,"cmfcont_lguts",fp)              
    
    # plot ConcFreqDur 3D
    for i,mc in enumerate(mcs):
        print("plot_ConcFreqDur",mc,component)
        plot_ConcFreqDur(fout,attr.key,mc,component,pecsw[i],threshold=0.00001,
                     pec_y=attr.pec_y)

    if attr.lguts == "true":

        days = depth[0][0].shape[0]/24
        for mc_id,mc in enumerate(mcs):
            for reach in attr.reaches:
                print("plot_results_overtime",mc,component,reach)
                # get data
                ind = list(reachlist["key"].values).index(reach)
               
                dep = depth[mc_id][ind]
                pec = pecsw[mc_id][ind]
                drift = spraydriftlist[(spraydriftlist["mc"]==mc)&(spraydriftlist["key"]==reach)]
                ind = list(lguts_reaches).index(reach)
                surv = survival[mc_id,ind,:]
                # make plot
                plot_reach_timeseries(fout,attr.key,mc,component,reach,t0,tn,
                                  daily,hourly,days,dep,pec,drift,surv)

            # plot survival
            if len(survival) == len(daily):
                xt = survival[0][0].shape[0]/(24*365)
            else:
                xt = survival[0][0].shape[0]/365
            
            if xt<1:
                surv =np.min(survival[mc_id],axis=1)
                print("map_vulnerability",mc,component,year)
                # prepare values for mapping
                res = pd.DataFrame({"surv":surv,"key":lguts_reaches})
                res = res.merge(reachlist,left_on="key", right_on="key", suffixes=('', ''))  
                res.sort_values(by="surv",inplace=True)  
                # make plot
                map_vulnerability(fout,attr.key,mc,component,year,res.surv.values,
                                  res.x.values,res.y.values,res.downstream_x.values,
                                  res.downstream_y.values,res)
            
            else:
                # surv = np.array([np.min(np.split(survival[mc_id][i],xt),axis=1) for i in range(survival[mc_id].shape[0])]).T
                def apply_func_annual(dat,year_ind,func=np.min):
                    """ """
                    return np.array([func(dat[i[0]:i[1]]) for i in year_ind])
                surv = np.array([apply_func_annual(survival[mc_id][i],
                                                   dat.year_ind,func=np.min) for i in range(survival[mc_id].shape[0])]).T
                
                
                for i,year in enumerate(range(t0.year,tn.year+1,1)):
                    print("map_vulnerability",mc,component,year)
                    # prepare values for mapping
                    res = pd.DataFrame({"surv":surv[i],"key":lguts_reaches})
                    res = res.merge(reachlist,left_on="key", right_on="key", suffixes=('', ''))  
                    res.sort_values(by="surv",inplace=True)  
                    # make plot
                    map_vulnerability(fout,attr.key,mc,component,year,res.surv.values,
                                      res.x.values,res.y.values,res.downstream_x.values,
                                      res.downstream_y.values,res)

FLAGS = None

if __name__ == "__main__":

    ###########################################################################
    # get command line arguments or use default
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath',type=str, default=get_fdir([""]))
    parser.add_argument('--zip',type=str, default="false")
    FLAGS, unparsed = parser.parse_known_args()    

    ##########################################################################
    # process data
    
    # read attributes xml-file
    attr = parse_info(os.path.join(FLAGS.fpath,"attributes.xml"))
    
    # read data
    dat = Data(FLAGS.fpath,attr.key,
               attr.cmfcont,attr.steps,attr.cascade,attr.lguts,
               attr.t0,attr.tn)
    
 
    # cmfcont
    if attr.cmfcont == 'true':
        component="cmfcont"
        print("report",component)
        make_plots(attr, FLAGS.fpath, dat.mcs,component,dat.t0,dat.tn,dat.hourly,dat.daily, 
                   dat.res[attr.cmfcont_pecsw][:],
                   get_twa(dat.res[attr.cmfcont_pecsw][:],window=7*24),
                   dat.res[attr.cmfcont_survival][:],
                   dat.res[attr.cmf_depth][:],
                   dat.reachlist,dat.lguts_reaches, dat.spraydriftlist)
    
    # steps
    if attr.steps == 'true':
        component="steps"
        print("report",component)
        make_plots(attr, FLAGS.fpath, dat.mcs,component,dat.t0,dat.tn,dat.hourly,dat.daily, 
                   dat.res[attr.steps_pecsw][:],
                   get_twa(dat.res[attr.steps_pecsw][:],window=7*24),
                   dat.res[attr.steps_survival][:],
                   dat.res[attr.cmf_depth][:],
                   dat.reachlist,dat.lguts_reaches, dat.spraydriftlist)    
    
    
    # cascade
    if attr.cascade == 'true':
        component="cascade"
        print("report",component)
        make_plots(attr, FLAGS.fpath, dat.mcs,component,dat.t0,dat.tn,dat.hourly,dat.daily, 
                   dat.res[attr.cascade_pecsw][:],
                   get_twa(dat.res[attr.cascade_pecsw][:],window=7*24),
                   dat.res[attr.cascade_survival][:],
                   dat.res[attr.cmf_depth][:],
                   dat.reachlist,dat.lguts_reaches, dat.spraydriftlist)    
#    # zip file
#    if FLAGS.zip == "true":
#        print("zip",os.path.join(FLAGS.fpath))
#        make_archive(os.path.join(FLAGS.fpath), 'zip', 
#                     os.sep.join(FLAGS.fpath.split("/")[:-1]))
    
        
    
    
    
    

    
    