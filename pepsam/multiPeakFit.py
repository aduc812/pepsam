from sage.calculus.calculus import var
from .pefit import peFit
import numpy as np
from lmfit import minimize, Parameters, Parameter
import quantities as pq
from sage.plot.plot import list_plot

from .utils import IntegerBank       
int_seq=IntegerBank()

gauss= lambda x,x0,a,FWHM: a*np.exp(-(x-x0)**np.float64(2)/(2*(FWHM/(2*np.sqrt(2*np.log(2))))**np.float64(2)))

class multiPeakFit(peFit):
    '''\
Fits the data with the sum of a set of peaks
'''

    def __init__(self, dataset, dataset_args={},
                 #xrange=[-np.infty,+np.infty], # this was intended to override  both xranges for prompt and dataset
                 **kwargs):
        '''\
Input:
    dataset - a peTab object containig dataset to be fit
    X,Y - an X and Y expressions to extract data from dataset
'''
        self.dataset = dataset
        self.dataset_kwargs = dataset_args
        try:
            (self.xCol,self.yCol,self.Weights)=dataset.get_table(**dataset_args)
        except ValueError:
            (self.xCol,self.yCol)=dataset.get_table(**dataset_args)
            self.Weights=None
        self.residual_function=self.multipeak_residual
        
    def multipeak_residual(self,pars,x, data=None):
        try:
            ngauss=pars['ngauss'].value
        except KeyError:
            ngauss=0    
            
        amplitudes=list()
        positions=list()
        FWHMs=list()
        for index in range(1,ngauss+1): # in text indexes start with 1
            amplparname='g_amplitude_'+repr(index)
            posparname='g_position_'+repr(index)
            widthparname='g_FWHM_'+repr(index)        
            amplitudes.append(pars[amplparname].value) # in amplitude and tau lists
            positions.append(pars[posparname].value)        # indexes start with zero
            FWHMs.append(pars[widthparname].value)
            
        bias = np.float(pars['bias'].value)    
        x=np.array(x)  # make sure we work with plain arrays    
        
        model=bias
        for idx in range(ngauss):
            model += gauss(x,positions[idx],amplitudes[idx],FWHMs[idx])
            
        if data is None:
            return model
        if self.Weights is None:
            return (model - data)
        return (model - data)*self.Weights
        
        
    def display_fit(self,result,**kwargs):
         if 'plotjoined' not in self.dataset_kwargs:
             self.dataset_kwargs['plotjoined'] = False
         if 'size' not in self.dataset_kwargs:
             self.dataset_kwargs['size'] = 1
         if 'scale' not in self.dataset_kwargs:
             self.dataset_kwargs['scale'] = 'linear'
         res_scale=kwargs.pop('res_scale','linear')
         (
            list_plot(
                   zip(
                       self.xCol,
                       self.residual_function(result.params,self.xCol)    
                       ),
                   plotjoined=True,color='red',
                   ymax=np.max(self.yCol), 
                   ymin=min(np.min(self.yCol),np.max(self.yCol)/100),
                   **kwargs)\
            + self.dataset.plot2d(**self.dataset_kwargs)
         ).show(figsize=[4,3]) 
         list_plot(zip(
                           self.xCol,
                           self.residual_function(result.params,self.xCol,self.yCol)
                       ),
                       size=self.dataset_kwargs['size']
                       ).show(figsize=[4,3],scale=res_scale)
         peFit.display_fit(self,result,**kwargs)
         
    def make_params(self,ngauss=1,print_mode=True):
        params = Parameters()
        params.add(name='bias', value=0, vary=True, )
        params.add('ngauss',value=ngauss,vary=False)
        for index in range(1,ngauss+1): # in text indexes start with 1
            params.add('g_amplitude_'+repr(index),value=1,vary=True,min=0)
            params.add('g_position_'+repr(index),value=0,vary=True,min=0)
            params.add('g_FWHM_'+repr(index),value=1,vary=True,min=0)
        if (print_mode is True):
            return self._print_formatted_parameters(params)
        else:
            return params
 
