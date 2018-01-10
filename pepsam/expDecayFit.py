from sage.calculus.calculus import var
from .pefit import peFit
import numpy as np
from lmfit import minimize, Parameters, Parameter
import quantities as pq
from sage.plot.plot import list_plot
#from sage.misc.html import html
#from .plotfast import plotfast 
from .utils import IntegerBank       
int_seq=IntegerBank()

# this type of step func is ok for numpy, but np.sign is not supported by quantities 
# 
_step=lambda x: np.ceil((np.sign(x)+1)/2.0)

# this is a quantities-proof ufunc for step function
#def _step(x):
#    '''\
#    this is a quantities-proof NaN-proof ufunc for step function
#    example:
#        _step((2,3,np.nan,-1,0)*pq.s)
#        array([ 1. ,  1. ,  nan,  0. ,  0.5]) * dimensionless
#'''
#    nan_idxs=np.isnan(x)
#    t=np.ceil((x/np.abs(x)+1)/2.0)
#    t[np.isnan(t)]=0.5
#    t[nan_idxs]=np.nan
#    return t

class expDecayFit(peFit):
    '''\
Fits the data with the convolution of prompt curve 
with multi-exponential function
if no prompt specified, a Dirac delta-function is assumed
(that is, the multi-exponential function is multiplied by step-function)
'''

    def __init__(self, dataset, dataset_args={},
                 prompt=None, prompt_args={}, conv_method='time',
                 #xrange=[-np.infty,+np.infty], # this was intended to override  both xranges for prompt and dataset
                 **kwargs):
        '''\
Input:
    dataset - a peTab object containig dataset to be fit
    X,Y - an X and Y expressions to extract data from dataset
    prompt - a peTab object containig prompt dataset
    promptX,promptY - an X and Y expressions to extract data from prompt
'''
        self.dataset = dataset
        self.dataset_kwargs = dataset_args
        
        try:
            (self.xCol,self.yCol,self.Weights)=map (np.array,dataset.get_table(**dataset_args))
        except ValueError:
            (self.xCol,self.yCol)=map (np.array,dataset.get_table(**dataset_args))
            self.Weights=None
        
        if (prompt is None):
            self.residual_function=self.multiexp_residual
        else:
            self.convmethod=conv_method
            self.prompt=prompt
            # and make sure we work with plain arrays 
            (self.prompt_raw_xCol,self.prompt_raw_yCol)=map (np.array,prompt.get_table(**prompt_args))
            self.residual_function=self.multiexpconv_residual
            self.promptx_begin=self.prompt_raw_xCol[0]
            # interpolate prompt at X values of dataset
            from scipy import interpolate
            self.prompt_interp_func = interpolate.interp1d(
                                    self.prompt_raw_xCol, self.prompt_raw_yCol,
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=0 # outside range prompt  
                                    )           #  is negligible
            self.normalize_prompt()
            
            xstep=np.diff(self.xCol)
            xstep=np.insert(xstep,0,xstep[0])
            if self.Weights is None:
                self.Weights=np.sqrt(xstep/xstep[0]) #1.0/np.sqrt(4e-6/(xstep/xstep[0])+np.abs(self.yCol)*0.01)
                
            range_steps=np.transpose(np.nonzero(abs(np.diff(xstep))>=xstep[0]/100))
            if len(range_steps)==0:
                self.hires_edge=len(self.xCol)
                self.full_hires=True
            else:
                self.hires_edge=int(range_steps[0])
                self.full_hires=False
    
    def normalize_prompt(self,pshift=0):
                                       
        self.prompt_yCol=self.prompt_interp_func(self.xCol-pshift)
        self.prompt_yCol/=np.sum(self.prompt_yCol)            
        ps=0
        pi=0
        pe=len(self.prompt_yCol)
        for i,p in enumerate(self.prompt_yCol):
            ps+=p
            if ps<=0.5:
                pi=i
            if ps>=1.0:
                pe=i
                break
        self.prompt_shortY=self.prompt_yCol[0:pe]
        self.prompt_barycenter=self.xCol[pi]
    
    def multiexpconv_residual(self,pars,x,data=None,):
        
        if 'xshift' in pars:       
            xshift = np.float(pars['xshift'].value)
        else:
            xshift = 0
        
        # prompt shift has a somewhat different meaning than xshift
        # xshift is not regarded in favor of pshift in convolution mode   
        if 'pshift' in pars:       
            pshift = np.float(pars['pshift'].value)
        else:
            pshift = 0
        
        if pshift != 0:
            self.normalize_prompt(pshift)        

            
        if 'ultrafast_amplitude' in pars:       
            ultrafast_amplitude = np.float(pars['ultrafast_amplitude'].value)
        else:
            ultrafast_amplitude = 0
            
        # transient absorption components    
        abs_amplitude=list()
        abs_tau=list()
        aamp_name='abs_amplitude_'
        atau_name='abs_tau_'
        for par in pars:
            if par.startswith(aamp_name): # found an abs parameter
                name=par[len(aamp_name):]
                if atau_name+name in pars:
                    a=pars[aamp_name+name].value
                    t=pars[atau_name+name].value
                else:
                    raise ValueError('absorption amplitude '+name+' with no matching tau')
                abs_amplitude.append(np.float(a))
                abs_tau.append(np.float(t))
                
        abs_coef = sum([A*np.exp(-(x[0:self.hires_edge]-x[0])/T) for A,T in zip(abs_amplitude,abs_tau) ]) 
                         
        IRF=self.multiexp_residual(pars,x[0:self.hires_edge], data=None, force_xshift = x[0],calc_abs=False)#self.prompt_barycenter)# xshift-self.prompt_barycenter)#+x[0])
        
        if self.convmethod=='fft':
            cvfix=np.real(np.fft.ifft(np.fft.fft(IRF[0:self.hires_edge])*np.fft.fft(self.prompt_yCol[0:self.hires_edge])))
        else:
            cvfix=np.convolve(self.prompt_shortY,IRF[0:self.hires_edge])[0:self.hires_edge]
        
        cvfix+=ultrafast_amplitude*self.prompt_yCol[0:self.hires_edge]
        cvfix*=np.float(10)**(-abs_coef)
        
        if self.full_hires:
            response=cvfix
        else:
            response=self.multiexp_residual(pars,x,data=None,force_xshift = self.prompt_barycenter+pshift,calc_abs=True)     
            response[0:self.hires_edge]=cvfix
            
        if data is None:
            return response
        if self.Weights is None:
            return (response - data)
        return (response - data)*self.Weights            
        
    def multiexp_residual(self,pars,x,data=None,force_xshift=None,force_bias=None,calc_abs=True):
        # unpack parameters:
        #  extract .value attribute for each parameter
        
        if force_bias is not None:
            bias=np.float(force_bias)
        elif 'bias' in pars:
            bias = np.float(pars['bias'].value)
        else:
            bias = 0
            
        if force_xshift is not None:
            xshift=np.float(force_xshift)        
        elif 'xshift' in pars:       
            xshift = np.float(pars['xshift'].value)
        else:
            xshift = 0
        
        amplitude=list()
        tau=list()
        rise=list()
        stretch=list()
        amp_name='amplitude_'
        tau_name='tau_'
        rise_name='rise_tau_'
        stretch_name='stretched_order_'
        
        # transient absorption components    
        abs_amplitude=list()
        abs_tau=list()
        aamp_name='abs_amplitude_'
        atau_name='abs_tau_'
        
        # hyperbolic components
        
        hyp_amp=list()
        hyp_tau=list()
        hyp_order=list()
        
        hyp_amp_name='hyp_amplitude_'
        hyp_tau_name='hyp_tau_'
        hyp_order_name='hyp_order_'
        
        for par in pars:
            if par.startswith(amp_name): # found an amp parameter
                name=par[len(amp_name):]
                if tau_name+name in pars:
                    if rise_name+name in pars:
                        r=pars[rise_name+name].value
                    else:
                        r=0
                    if stretch_name+name in pars:
                        beta=pars[stretch_name+name].value
                    else:
                        beta=1
                    a=pars[amp_name+name].value
                    t=pars[tau_name+name].value
                else:
                    raise ValueError('amplitude '+name+' with no matching tau')
                amplitude.append(np.float(a))
                tau.append(np.float(t))
                rise.append(np.float(r))
                stretch.append(np.float(beta))
                
            if par.startswith(aamp_name): # found an abs parameter
                name=par[len(aamp_name):]
                if atau_name+name in pars:
                    a=pars[aamp_name+name].value
                    t=pars[atau_name+name].value
                else:
                    raise ValueError('absorption amplitude '+name+' with no matching tau')
                abs_amplitude.append(np.float(a))
                abs_tau.append(np.float(t))
                
            if par.startswith(hyp_amp_name): # found an abs parameter
                name=par[len(hyp_amp_name):]
                if hyp_order_name+name in pars:
                    a=pars[hyp_amp_name+name].value
                    o=pars[hyp_order_name+name].value
                    t=pars[hyp_tau_name+name].value
                else:
                    raise ValueError('hyperbolic amplitude '+name+' with no matching order or tau')
                hyp_amp.append(np.float(a))
                hyp_order.append(np.float(o))
                hyp_tau.append(np.float(t))
                
        x=np.array(x)  # make sure we work with plain arrays
        if calc_abs:
            abs_coef = sum([A*np.exp(-(x-xshift)/T) for A,T in zip(abs_amplitude,abs_tau) ]) 
        else:
            abs_coef = 0
            
        model = np.zeros(x.shape) + sum([A/(1+x/T)**O for A,O,T in zip(hyp_amp,hyp_order,hyp_tau) ]) 
        for A,T,R,beta in zip(amplitude,tau,rise,stretch):
            if R == 0:
                model+= A*np.exp(-((x-xshift)/T)**beta)*_step(x-xshift)
            else:
                model+= A*(1-np.exp(-(x-xshift)/R))*np.exp(-((x-xshift)/T)**beta)*_step(x-xshift) 
            
        model*=np.float(10)**(-abs_coef)
        model+= bias   
            
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
             self.dataset_kwargs['scale'] = 'semilogy'            
         fname_com=kwargs.pop('filename',None)
         res_scale=kwargs.pop('res_scale','linear')
         scale=kwargs.pop('scale','semilogy')
         
         fname_plot = None if fname_com is None else 'plot_' + fname_com
         fname_res  = None if fname_com is None else 'res_' + fname_com
         #from sage.repl.rich_output import pretty_print 
         #pretty_print(html('<table><tr><td>'))
         (
            list_plot(
                   zip(
                       self.xCol,
                       self.residual_function(result.params,self.xCol)    
                       ),
                   plotjoined=True,color='red',
                   **kwargs)\
            + self.dataset.plot2d(**self.dataset_kwargs)
         ).show(#filename=fname_plot,
                figsize=[6,3],
                ymax=np.max(self.yCol)*1.1,
                ymin=min(np.min(self.yCol),np.max(self.yCol)/100),
                scale=scale
                ) 
         resid_plt=list_plot(zip(
                           self.xCol-self.xCol[0]+self.xCol[1],
                           self.residual_function(result.params,self.xCol,self.yCol)
                       ),
                       size=self.dataset_kwargs['size'], color='blue',
                       )
         resid_plt.show(#filename=fname_res,
                        figsize=[6,3],scale=res_scale)
         #plotfast(resid_plt,scale=res_scale)
         #if fname_com is not None:
           #  pretty_print(html('<table><tr><td>')) 
             #html('<img src="'+fname_plot+'">')
           #  pretty_print(html('</td></tr><tr><td>'))
             #html('<img src="'+fname_res+'">')
           #  pretty_print(html('</td></tr></table>'))
          #   pretty_print(html('</td><td>'))
         peFit.display_fit(self,result,**kwargs)
         #pretty_print(html('</td></tr></table>'))
         
    def make_params(self,nexp=3,ultrafast=False,print_mode=True):
        params = Parameters()
        params.add('bias', value=0, min=0)
              
        if hasattr(self,'prompt'):   
            params.add('ultrafast_amplitude', value=0, vary=False)
            params.add('pshift', value=0, vary=0) 
        else:
            params.add('xshift', value=0, vary=0)  
            
        for i in range(1,nexp+1):
            amplitude=np.float(np.max(self.yCol)/2)
            params.add('amplitude_'+repr(i),amplitude,min=0)
            minscale=np.min(self.xCol)
            maxscale=np.max(self.xCol)
            tau=np.float(minscale+(maxscale-minscale)/10/float(nexp)*i)
            params.add('tau_'+repr(i),tau,min=0)
        if (print_mode is True):
            return self._print_formatted_parameters(params)
        else:
            return params
        
