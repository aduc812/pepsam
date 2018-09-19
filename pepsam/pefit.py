
# -*- coding: utf-8 -*-
from lmfit import minimize, Parameters, Parameter
import numpy as np
import quantities as pq

from matplotlib.figure import Figure
#from matplotlib.pyplot import figure as Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import rcParams

from .utils import pretty_str_units, test_ipython

def copy_pars(params):
    '''\
    Copies Parameters object.
    The Parameters object passed to a fitting function gets filled by fitting data and is not useable for future fits.
    Use this function to create fresh copy of it.
    '''
    cpy_params= Parameters() 
    for name,param in params.items():         
        cpy_params.add(value=param.value, name=param.name, vary=param.vary, expr=param.expr, min=param.min, max=param.max)
    return cpy_params
    



class peFit(object):
    '''\
An abstract class for fitting 2-D data 
uses lmfit module http://pypi.python.org/pypi/lmfit/
homepage http://cars9.uchicago.edu/software/python/lmfit/

The functions to be overridden by specific fitting model:
__init__(self,*args,**kwargs)
residual_function(self,x,parameters,data=None)
make_params(self,*args,**kwargs)
display_fit(self,result)
'''
    def __init__(self,*args,**kwargs):
        '''\
Actual fiting model classes have to override __init__ to accept all 
pre-defined data used in residual_function() to calculate residuals,
such as experimental curve to be fit, any prompt functions and so on.
As minimum, defines self.Xcol and self.Ycol atributes, which contain 
data columns (numpy array-like objects) to be fitted.
'''
        self.Xcol=None
        self.Ycol=None
        self.Weights=None

    def residual_function(self,parameters,x,data=None):
        '''\
This function defines residual calculation method. 
Input:
    x - independent variable value (a numpy array of values)
    Parameters object for a model.
    data - the data being fitted (can be), corresponding to value of x 
Does not take any additional data, uses those stored in the class attributes during __init__() 
If data==None, evaluates model function instead of giving residual
Returns numerical value - residual calculated on x
'''
    pass

    def make_params(self,print_mode=True,*args,**kwargs):
        '''\
This function should generate and return sample Parameters object (initial guess) to be edited by the user
before actual fitting. May take any parameters required, but should specify all defaults.
if print_mode=True, returns the string containing structure like the following :
    
params = Parameters()
params.add('nexp',value=3,vary=false)
params.add('amplitude_1', value=607, min=0)
params.add('tau_1', value=0.5,min=0)
params.add('amplitude_2', value=20,min=0)
params.add('tau_2', value=12, min=0)
params.add('amplitude_3', value=20,min=0)
params.add('tau_3', value=5,min=2)
params.add('bias', value=1, min=0.0001,)
params.add('xshift', value=2.9, min=0,max=5, vary=0)    
    
so that eval(self.make_params(...)) creates params structure in memory.    
    
'''
    pass


    def fit(self,initial_guess,display=True,displayargs={},**kwargs):
        '''\ 
This finction actually fits the data.
takes an initial_guess Parameters object
returns the Parameters object 
'''
        result = minimize(self.residual_function, initial_guess, args=(self.xCol,self.yCol),**kwargs)
        if (display==True) :
            self.display_fit(result,**displayargs)
        return result

    def display_fit(self,result,**kwargs):
        '''\
This function is intended to plot the initial data and model curve,
and pretty-print the fitting parametrs.
Takes the fitting result (Parameters object)
returns nothing.
It can be overridden for specific output.
'''
        if(result.success):
            print 'Fit succeeded'
        else:
            print 'Fit failed, try redefifing initial guess'
        if(result.errorbars is False):
            print 'Errors not estimated, you may have overparametrized the function' 
        print 'Chi-sqr: ' + str(result.chisqr)  
        for name, par in result.params.items():
            if (result.errorbars is True):
                print '  %s = %.4g err: %.4g'% (name, par.value, par.stderr)
            else:
                print '  %s = %.4g'% (name, par.value)
            
    def _print_formatted_parameter(self,parameter):
        if isinstance(parameter, Parameter) == False:
            raise TypeError('_print_formatted_parameter() takes only lmfit.parameter.Parameter objects')
            
        outp='params.add('
        attrlist=('name','value','min','max','vary','expr')
        
        for attr in attrlist:
            if parameter.__getattribute__(attr) is not None:
                outp+=  attr + '=' + repr(parameter.__getattribute__(attr)) + ', '
        return outp+')'
    def _print_formatted_parameters(self,parameters):
        if isinstance(parameters, Parameters) == False:
            raise TypeError('_print_formatted_parameters() takes only lmfit.parameter.Parameters objects')
            
        outp='params = Parameters()\n'
        for (name,param) in parameters.items():
            outp+=self._print_formatted_parameter(param)+'\n'
        return outp
#########################################################################################        
# multiple curves fitting
#########################################################################################        
        
        
class MultifitResult(object):
    
    def __init__(self, Xcol=None, results_list=None,res_len=0):
        if (Xcol is not None):
            self.xcol=Xcol
        else: 
            self.xcol=[None for i in range(res_len)]
        if (results_list is not None):
            self.results_list=results_list
        else: 
            self.results_list=[None for i in range(res_len)]
               
        object.__init__(self)
        #print len (self.xcol) # DEBUG
        
    def append(self,x,result,pos=None):
        if pos is None:
            self.xcol.append(x)
            self.results_list.append(result)
        else:
            self.xcol[pos]=x
            self.results_list[pos]=result
            
    def get_param_list(self):
        return self.results_list[0].params.keys()
 
    def get_result_col(self,parname):
        yval=list()
        for i,fit in enumerate(self.results_list):
            try:
                yval.append(fit.params[parname].value)
            except KeyError:
                yval.append(np.nan)
        return np.array(yval)
    
    def plot(self,parname,**kwargs):
        '''\
        keywords:
        errorbar - shows errorbars in fit, true by default 
        legend - shows legend, true by default 
        other kywords are passed to Figure.add_axes()
        Most important ones:
        xscale, yscale - [‘linear’ | ‘log’ | ‘symlog’]
        xlim, ylim - length 2 sequence of floats - to determine plot bounds'''

        showerr=kwargs.pop('errorbar',True)
        showleg=kwargs.pop('legend',True)
        
        figure = Figure(figsize=[7,5])
        rect=(
                rcParams['figure.subplot.left'],
                rcParams['figure.subplot.bottom'],
                rcParams['figure.subplot.right']-rcParams['figure.subplot.left'],
                rcParams[ 'figure.subplot.top']-rcParams['figure.subplot.bottom']
             )
        figure.suptitle(parname)    
        main_plot = figure.add_axes(rect,**kwargs)
        main_plot.set_xlabel(pretty_str_units(pq.Quantity(self.xcol[0]).units))
        main_plot.set_ylabel(parname)
        
        x=[]
        redx=[]
        yelx=[]
        y =[]
        redy=[]
        yely=[]
        ex = None
        ey =[]
        
        if parname=='chisqr':
            getval=lambda fit,pname: fit.chisqr
            geterr=lambda fit,pname: 0
        else:
            getval=lambda fit,pname: fit.params[pname].value
            geterr=lambda fit,pname: fit.params[parname].stderr
            
        for i,fit in enumerate(self.results_list):
            try:
                xypair=(np.float64(self.xcol[i]),getval(fit,parname),geterr(fit,parname))
            except KeyError:
                continue
            if fit.success is False:
                yelx.append(xypair[0])
                yely.append(xypair[1])                 
            elif not hasattr(fit,'covar'):
                redx.append(xypair[0])
                redy.append(xypair[1])   
            elif fit.covar is None:
                redx.append(xypair[0])
                redy.append(xypair[1])
            else:
                x.append(xypair[0])
                y.append(xypair[1])
                ey.append(xypair[2])
                
        if not showerr:
            ey=None
                
        main_plot.errorbar(x, y,
                      ey,ex,  
                      fmt='.',
                      color='blue', 
                      ecolor='orange',
                      label="good fit",
                      )
                        
        main_plot.errorbar(redx, redy,
                      None,None,  
                      fmt='.',
                      color='red', 
                      ecolor='orange',
                      label="overparametrized",
                      )
        main_plot.errorbar(yelx, yely,
                      None,None,  
                      fmt='.',
                      color='yellow', 
                      ecolor='orange',
                      label="not succeeded",
                      )                
        if showleg :                     
            main_plot.legend(fancybox=True)
        figure.set_canvas(FigureCanvasAgg(figure)) 
        
        if test_ipython():
            #print 'ipython'
            import tempfile
            tmpimg=tempfile.NamedTemporaryFile(suffix='.png',delete=True)
            figure.savefig(tmpimg.name,format='png') 
            from IPython.display import Image, display
            display(Image(filename=tmpimg.name))        
        else: 
            #print ' no ipython'
            figure.savefig(parname+'.svg')
            
        #figure.show()
        #figure.close()
#    def py_plot(self,parname):
#        import matplotlib.pyplot as plt 
#        import numpy as np    
#        p = plt.figure(figsize=(7,5))
#        plt.xlabel(pretty_str_units(pq.Quantity(self.xcol[0]).units)) 
#        plt.ylabel(parname) 
#        plt.title(parname)
#        plt.grid(True)
#        x=[]
#        redx=[]
#        yelx=[]
#        y =[]
#        redy=[]
#        yely=[]
#        ex = None
#        ey =[]
#        
#        for i,fit in enumerate(self.results_list):
#            try:
#                xypair=(self.xcol[i],fit.params[parname].value,fit.params[parname].stderr)
#            except KeyError:
#                continue
#            if fit.success is False:
#                yelx.append(xypair[0])
#                yely.append(xypair[1])                
#                
#            elif fit.covar is None:
#                redx.append(xypair[0])
#                redy.append(xypair[1])
#            else:
#                x.append(xypair[0])
#                y.append(xypair[1])
#                ey.append(xypair[2])
#                    
#        plt.errorbar(x, y,
#                      ey,ex,  
#                      fmt='.',
#                      color='blue', 
#                      ecolor='orange',
#                      label="good fit")
#                        
#        plt.errorbar(redx, redy,
#                      None,None,  
#                      fmt='.',
#                      color='red', 
#                      ecolor='orange',
#                      label="overparametrized")
#        plt.errorbar(yelx, yely,
#                      None,None,  
#                      fmt='.',
#                      color='yellow', 
#                      ecolor='orange',
#                      label="not succeeded")                
                             
#        plt.legend(fancybox=True, loc=0)
#        plt.savefig(parname+'.svg') 
#        plt.clf()  
    def make_report(self,filename=None):
        parlist=self.get_param_list()
        report='X\t'
        report_arr=np.array(self.xcol).flatten()
        for param in parlist:
            report+=(str(param))+'\t'
            report_arr=np.vstack((report_arr,self.get_result_col(param)))
        report+='\n'
        for i in report_arr.transpose():
            for j in i:
                report+=str(j)+'\t'
            report+='\n'
            
        if filename is None:
            print report
        else:
            with open(filename,'w') as f:
                f.write(report)
                
        
class peMultiFit (object):
    def __init__(self,fit_method,method_kwdlist,init_parameters,func_to_get_fit_id = lambda i,kwds:i,import_result=None,import_param_names={},ranges_dict={},out_of_range_vals={}):
        self.fit_method=fit_method
        self.method_kwdlist=list(method_kwdlist)
        self.init_parameters=init_parameters
        self.result=MultifitResult(res_len=len(self.method_kwdlist))
        self.func_to_get_fit_id=func_to_get_fit_id
        object.__init__(self)
        self.import_result=import_result
        self.import_param_names=import_param_names
        self.ranges_dict=ranges_dict
        self.out_of_range_vals=out_of_range_vals
        
    def fit_one(self,i,prev_params,**kwargs) :
    
        showprogress=kwargs.pop('showprogress',True)
        displayfit=kwargs.pop('displayfit',False)
    
        param_set=self.method_kwdlist[i]
        fit_id=self.func_to_get_fit_id(i,param_set)
        
        # import fixed params from previous layer 
        if not isinstance(self.import_param_names,dict):
            imp_params=dict(zip(self.import_param_names,[False for j in self.import_param_names]))
        else:
            imp_params=self.import_param_names
            
        for name,vary in imp_params.items():
            param=self.import_result.results_list[i].params[name]
            prev_params.add(value=param.value, name=param.name, vary=vary, expr=param.expr, min=param.min, max=param.max)
                          
        # check if we fall in certain range
        for key,value in  self.ranges_dict.items():
            if key in prev_params:
                if not(min(value)<=fit_id<=max(value)):
                    prev_params[key].value=self.out_of_range_vals[key] if key in self.out_of_range_vals else 0
                    prev_params[key].vary=False
                else: # do not forget to switch varying back on 
                    if (prev_params[key].vary==False):
                        prev_params[key].value=self.init_parameters[key].value
                        prev_params[key].vary=self.init_parameters[key].vary
            
        # print progress
        if showprogress:
            print 'fit %d of %d: %s' % (i, len(self.method_kwdlist),fit_id)
        cvfitter=self.fit_method(**param_set)
        displayargs=kwargs.pop('displayargs',{})
        displayargs.update(dict(filename='fit'+str(i)+'.png',))
        kwargs['displayargs']=displayargs
        this_result=cvfitter.fit(prev_params,display=displayfit,**kwargs)
        
        self.result.append(fit_id,this_result,i)
            
        #prepare params for next fit:
        next_params= Parameters()
        for name,param in prev_params.items():         
            next_params.add(value=param.value, name=param.name, vary=param.vary, expr=param.expr, min=param.min, max=param.max)        
        return next_params


            
    def fit(self,startpoint=0,**kwargs):
        prev_params= copy_pars(self.init_parameters)
            
        first_params=self.fit_one(startpoint,prev_params,**kwargs)    
        prev_params= copy_pars(first_params) 
       
        for i in range(startpoint+1,len(self.method_kwdlist)): 
            prev_params=self.fit_one(i,prev_params,**kwargs)
            
        prev_params= copy_pars(first_params)     
        for i in reversed(range(0,startpoint)): 
            prev_params=self.fit_one(i,prev_params,**kwargs)
            
    def plot_result(self,**kwargs):
        if self.result is not None:
            self.result.plot('chisqr',**kwargs)  
            for parameter in self.get_variable_param_list():
                self.result.plot(parameter,**kwargs)  
                
    def get_variable_param_list(self):
        parlist=[]
        for key,val in self.init_parameters.items():
            if val.vary:
                parlist.append(key)
        return parlist
