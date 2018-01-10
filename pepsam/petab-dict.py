# encoding: utf-8
###############################################################################
# imports section

## pemongo imports
from .mongo_connection import mongoConnection, petabTable
from .filters import filter_wrapper
from .expr_set import Expr_set, E, _expr_set_from_dict, ESET_KEY
from .utils import ColorBank, unitdict, units_and_array,\
            pretty_repr_units,pretty_str_units,\
            _quantity_to_dict, _quantity_from_dict,\
            _expression_to_dict, _expression_from_dict,\
            dev_dims, extract_multivar_args
from .pescript import peScriptRef

## pymongo imports
from bson import ObjectId

## numpy
import numpy as np

## physical quantities
import quantities as pq

##sage imports
#import sage

from sage.symbolic.expression import Expression as sage_expr
#import sage.ext.fast_callable.fast_callable
#import sage.plot.graphics.Graphics
#from sage.misc.sage_eval import sage_eval
from sage.calculus.calculus import var

## other tools
import math
import datetime
###############################################################################

EXPRESSIONS_COMPARE_PRECISION=1E-10



glob_color=ColorBank()




class peTab(dict,
#            sage.structure.sage_object.SageObject
            ):
    '''
    peTab objects are dictionaries of values,
    representing physical experiment data
    Normally they are not created directly, but rather
    generated by peSet objects or fetched from mongoDB database.
'''

    def __init__(self,*args,**kwargs):
#       sage.structure.sage_object.SageObject.__init__(self)
        if isinstance(args[0],dict):
            self._init_from_dict(args[0])           
        elif isinstance(args[0],ObjectId):# fetch from database
            self._init_from_OID(args[0])
        elif isinstance(args[0],basestring):# fetch from database using string OID
            self._init_from_OID(ObjectId(args[0]))
        else:
            raise NotImplemented('cannot create peTab from type "' + str(type(args[0]))+'"' )
            
    def _init_from_OID(self,oid):
        '''\
Loads a peTab object from DB into a valid peTab instance
'''
        dict_=petabTable.find_one({"_id": oid})
        if dict_==None:
            raise ValueError('cannot find OID ' + repr(oid)+ ' in DB')
        self._init_from_dict(dict_)    
    
    
    def _init_from_dict(self,dict_):
        '''
Converts a dictionary (manually created or loaded from DB) into a valid peTab instance
'''   
        #set very defauls - may be overridden in dict_, but should exist
        self['defX']=None
        self['defY']=None
        self['defZ']=None
        self['_default_axis']=list()
        self['_params_to_show']=list()
        
        for key,value in dict_.items():
             if isinstance(value,dict):
                if value.has_key('value') :
                    if value.has_key('units') :
                        self[key]=_quantity_from_dict(value)
                    else:
                        self[key]=np.array(value['value'])
                elif value.has_key('expression') and value.has_key('arguments'):
                    self[key]=_expression_from_dict(value)
                elif value.has_key(ESET_KEY):
                    self[key]=_expr_set_from_dict(value)
                elif value.has_key('oid') and value.has_key('name'):
                    self[key]=peScriptRef(oid=value['oid'],name=value['name'])
             elif isinstance(value,tuple):
                self[key]=list(value) # in db only lists supported
             else:
                self[key]=value
                                
    def dict(self):
        '''
Creates plain python dictionary from peValue instance. The dictionary can be loaded into mongoDB. 
'''   
        adict={}
        for key,value in self.items():
            if isinstance (value,pq.quantity.Quantity):
                adict[key]=_quantity_to_dict(value)
            elif isinstance (value,sage_expr):
                adict[key]=_expression_to_dict(value)
            elif isinstance(value,np.ndarray):
                if value.dtype!=np.dtype('object'):
                    adict[key]={'value':value.tolist()}
            elif isinstance (value,Expr_set):
                adict[key]=value.dict()        
            else:
                adict[key]=value
        return adict
                 
    def db(self):
        '''\
save to database. Adds to peTab instance an '_id' key with bson.ObjectID from database.
'''
        if self.has_key('_id'):
            raise RuntimeError('was already saved under OID '+repr (self['_id'])+'. Use peTab.updatedb instead')
        # check if we are not inserting the existing thing
        data_to_insert=self.dict() 
        existing_things=petabTable.find(data_to_insert)
        if existing_things.count()==0:
            with mongoConnection.start_request():
                oid=petabTable.insert(data_to_insert)
                self['_id']=oid
                return self['_id']
        else:
            raise RuntimeError('The record(s) with exactly the same fields already exist (ex. ' + repr(existing_things.next()['_id']) + '). If you want to update it, please do it explicitly')
    
    id=lambda self: self[_id]
    
    def updatedb(self):
        '''
        updates the record in database. Use after the peTab object was altered, to save the changes.
        Use with extreme care. 
        '''
        if not self.has_key('_id'):
            raise RuntimeError('This peTab instance was not ever saved into DB. Use peTab.db() to do it')
        existing_record=petabTable.find_one({'_id':self['_id']},fields=['_id'])
        if not existing_record:
            raise RuntimeError('This peTab instance has an _id which is not found in database')
        petabTable.save(self.dict())
############################################################################################ 
# copying and comparsion, mainly for testing purposes  
############################################################################################ 
    def copy(self):
        '''\
returns not-so-shallow copy of a peTab, 
by filling an empty peTab with
copies of mutable elements having copy() method,
copies of lists using newlist=list(oldlist)
and all other elements themselves.
note that usually mutable elements provide shallow copy(),
their nested objects are not copied but rater mapped.
This should not be an issue in suggested way of peTab usage.
                           
'''
        obj=peTab.__new__(peTab)
        for key,value in self.items():
            if hasattr(value,'copy'):
                obj[key]=value.copy()
            elif isinstance(value,list): # lists do not have copy()
                obj[key]=list(value)
            else:
                obj[key]=value
        return obj
          
    def __eq__(self,other):
        '''\
Compares two peTab objects
as numpy arrays perform an element-wise comparision and then 
expect using array.any() or array.all(),
we have to redefine __eq__ and __ne__ to be able to compare peTabs containing numpy arrays
'''

        if self is other:
            return True
        if not isinstance(other,peTab):
            return False
        if not len(self.keys())==len(other.keys()):
            return False
            
        for key in self:
            if not (key in other):
                return False
                
            if isinstance(self[key],np.ndarray) :
                if not isinstance(other[key],np.ndarray):
                    return False
                if not self[key].shape==other[key].shape:
                    return False
                if not (self[key]==other[key]).all():
                    return False
            elif isinstance(self[key],sage_expr):
                if not isinstance(other[key],sage_expr):
                    return False
                if not (self[key]==other[key]):
                    # constants from expressions are printed into repr() with limited precision.
                    # we treat them equal anyway
                        try:
                            expr=(self[key]-other[key])/(self[key]+other[key])
                            num=abs(expr.N())
                        except TypeError:
                            return False
                        if not num<EXPRESSIONS_COMPARE_PRECISION:
                            return False
                 
            else: # simply compare instances
                if not self[key]==other[key]:
                    return False
            
        return True
        
    def __ne__(self,other):
        ''' see peTab.__eq__'''
        return not self.__eq__(other)    
############################################################################################
#
#   main peTab functionality: evaluation of expressions,
#   quiering for data, interpolation and plotting
#
############################################################################################           
    def _expand_expression(self,expression):
        '''\
Expands an expression, replacing in it parameters-expressions with the corresponding expressions
Argument: an instance of sage.symbolic.expression.Expression 
Output: expanded expression, so that none of its arguments correspond to an expression 
in this peTab instance
'''
        #if not isinstance(expression,sage.symbolic.expression.Expression):
        #    return expression # no arguments - job may have been finished
        if isinstance(expression,sage_expr):
            expression=Expr_set((expression,))
        #try:
        args=expression.arguments()
        #except AttributeError:
        #    return expression # not an expression-use as is
        valdict={} # a set of expression -parameters to evaluate
        for arg in args:
            str_arg=repr(arg)
            if (str_arg in self): # found 'our' argument
                val=self[str_arg]
                if isinstance(val,(sage_expr,Expr_set)): 
                    if isinstance(val,sage_expr):
                        val=Expr_set((val,))
                    valdict.update({str_arg:self._expand_expression(val)})
                if isinstance(val,peScriptRef):
                    if isinstance(val.value,(sage_expr,Expr_set)):
                        valdict.update({str_arg:self._expand_expression(val.value)})

        return expression(**valdict)
##########################################################################################       
    def _eval_single_expr(self, expression):
        '''Evaluate one single expanded expression'''
        if not isinstance(expression,sage_expr):
            raise TypeError('_eval_single_expr takes only single expressions')
        arglist=[]
        vallist=[]
        args=expression.arguments()
        #print expression # DEBUG
        # loop over arguments
        found=False # an indicator that unresolved peTab-related arguments are found
        for arg in args:
            str_arg=repr(arg)
            #print repr(str_arg) # DEBUG
            if str_arg in self: # found 'our' argument
                found=True
                arglist.append(arg)
                if isinstance(self[str_arg],peScriptRef):
                    vallist.append(self[str_arg].value)
                else:
                    vallist.append(self[str_arg])
            elif str_arg in unitdict: # found a unit, which is not overridden by parameter name
                found=True
                arglist.append(arg)
                vallist.append(unitdict[str_arg])
            else: # found unknown argument, leave it as is
                arglist.append(arg)
                vallist.append(arg)
        #print arglist  #DEBUG
        # check if we have found 'our' unresolved arguments
        #print arglist #DEBUG
        #print vallist #DEBUG
        if (found):
            from sage.ext.fast_callable import fast_callable
            pExpr=fast_callable(expression,vars=arglist)
            #print pExpr(*(vallist)) #debug
            return pExpr(*(vallist))
        else: # no unresolved arguments - the end
            return expression  
            
###############################################################################                         
    def eval_expr(self,expression):
        ''' \
This function evaluates peTab-related expressions, replacing the variable by corresponding 
parameter value, i.e. var('parameter_name') -> peTab['parameter_name']
Is  recursive, as the parameter value may be itself an expression
!!!! Beware of circular links in parameters-expressions !!!!
'''      
        
        # check if we have string - treat as a variable with that name
        if isinstance(expression, basestring): 
            expression=var(expression)
        
        #expand the expression        
        expression=self._expand_expression(expression)
            
        
        try: #check if we have a collection of expressions
            expression_iterator = iter(expression)
        except TypeError: #this is a single expression           
            return self._eval_single_expr(expression)
        else: # expression is iterable
            if len(expression)==1: # treat iterable with one element as this element
                return self._eval_single_expr(expression[0])
            return map(self._eval_single_expr,expression)
            
################################################################################################           



    def _get_table(self,*args,**keywords):
        '''\
This function returns a tuple of values(arrays), representing
the X and Y columns of a table. The values are calculated using given expressions
for  X, Y (and Z if applicable). If either (or both) are not given, the expressions for defX and defY are used.
Yexpr here should be a single expression.

Arguments:
    an iterable (X,Y,Z,....) containing an expressions for the axis defined.
    All expressions but the last one should evaluate to a vector-like arrays in different dimensions 
    The last one is considered as an independent variable, and should evaluate to an array 
    with (len(iterable)-1) dimensions. If this argument is present, the X,Y,and Z keywords are ignored.
    
Keywords:
    X,Y,Z
    an easier way to understand get_table is to use X,Y, and Z keywords.    
'''
        expr_list, keywords=extract_multivar_args(
            args,keywords,
            single_keywords=('X','Y','Z'),
            single_kwd_defaults=(self['defX'],self['defY'],self['defZ']),
            compound_keyword='expr_set')
        # remove None - crutial for 2d datasets, which have defZ=None
        while(True):
            try:
                expr_list.remove(None)
            except (ValueError,AttributeError):
 ## there are no more Nans  or  expr_list is already Expr_set and does not have Nones and remove()
                break       
        # if expr_set is given by an explicit kwd:    
        expr_set=E(expr_list)    
            
        eval_at=keywords.pop('eval_at',None)        
        #eval_at ((expr,value[,type]),...)
    #        (expr,value,'coerce') -- default
    #        (expr,(value,tolerance),'vicinity_avg')
    #        (expr,(from,to),'range_avg')
    #        (expr,(from,to),'range_sum')
    #        (expr,num,'seq_num') 
        
        range_x = keywords.pop('xrange', None)
        range_y = keywords.pop('yrange', None)
        var_ranges = keywords.pop('var_ranges', (range_x, range_y))       
        deficit=len(expr_set)-len(var_ranges)
        if deficit>0:
            var_ranges+=(None,)*deficit
        
        units_x = keywords.pop('xunits',None)
        units_y = keywords.pop('yunits',None)
        units_z = keywords.pop('zunits',None)
        var_units = keywords.pop('var_units',(units_x, units_y, units_z))
        deficit=len(expr_set)-len(var_units)
        if deficit>0:
            var_units+=(None,)*deficit
      
        var_set=map(self.eval_expr,expr_set) # we keep var_set as a list, even if it contains 1 element 
        
        ## A universal selection rule ????       
        ## evaluate  "eval_at" variables and rules 
        if(eval_at is not None):
            eval_vars=self.eval_expr(Expr_set([ea[0] for ea in eval_at]))
            if not isinstance(eval_vars,list): 
                eval_vars=[eval_vars]
            
            eval_conditions=[ea[1] for ea in eval_at]
            eval_methods=[]
            for cond in eval_at:
                if len(cond)<3:
                    meth='coerce'
                else:
                    meth=cond[2]
                eval_methods.append(meth)
            #print eval_vars #DEBUG
        else:
            eval_vars=[]    
            
        ## sort the array along every selected dimention
        argsort_1d =lambda y:np.argsort(y,axis=dev_dims(y)[0]).squeeze(axis=range(0,len(y.shape)).remove(dev_dims(y)[0]))
        sorted_dims=[]
        for i in eval_vars+var_set: # first sort on eval_vars, as we will then maybe select ranges from them
            dims=dev_dims(i)
            #print dims #DEBUG
            if len(dims)!=1:
                continue  # sort only 1d arrays
            dim=dims[0]
            if dim in sorted_dims:
                continue # sort once only along every dimension
            sorted_dims.append(dim)
            # get index by sorting and then squeezing along all dims but dim, which is the only non-degenerate
            index= np.argsort(i,axis=dim).squeeze(axis=range(0,len(i.shape)).remove(dim))
            # sort everything using that index        
            for idx,j in enumerate(var_set):  
                if dim in dev_dims(j):    # sort along non-degenerate dimension dim
                    var_set[idx]=j.take(index,axis=dim)                    
            for idx,j in enumerate(eval_vars):  
                if dim in dev_dims(j):    # sort along non-degenerate dimension dim
                    eval_vars[idx]=j.take(index,axis=dim)
            
        # extract desired ranges
        for var,var_range in  zip(var_set,var_ranges):
            if var_range is None: # all range
                continue
            dims=dev_dims(var)
            if len(dims)!=1: # no range option for non-1D arrays
                continue
            dim=dims[0]    
            range_start,range_end=sorted(var_range)
            #print range_start,range_end #DEBUG
            condition=np.logical_and(range_start<var,var<range_end).flat
            #print [ i for i in condition] #DEBUG
            for j,var in enumerate(var_set):                  
                if var.shape[dim]>1:
                    var_set[j]=var.compress(condition, axis=dim)          
            for j,var in enumerate(eval_vars):      
                if var.shape[dim]>1:
                    eval_vars[j]=var.compress(condition, axis=dim) 
        ## contract the arrays according to 'eval_at' rules    
        if(eval_at is not None):    
            for i in range(0,len(eval_at)):  # could be while(True), but this is a bit safer 
                
                try:
                    (expr_val,desired_val,rule)=(eval_vars.pop(),eval_conditions.pop(),eval_methods.pop())
                except IndexError: # end the loop on no elements left
                    break
                    
                dims=dev_dims(expr_val)
                
                #print rule #DEBUG
                if len(dims)==0:
                    continue
                
                elif rule=='seqnum':        
                    if len(dims)>1:
                        raise ValueError('\'seqnum\' option works only for 1-D variables')
                    condition=np.zeros(expr_val.shape[dims[0]])
                    #print len(condition)
                    if len(condition)<=desired_val:
                        raise IndexError('index '+repr(desired_val)+ ' is too large, of total ' + repr(len(condition)))
                    condition[desired_val]=1
                    conditions=(condition,)
                    
                elif rule=='coerce':
                    if len(dims)>1:
                        raise ValueError('\'coerce\' option works only for 1-D variables')
                    distances=abs(expr_val-np.ones(expr_val.shape)*desired_val)
                    condition=np.zeros(expr_val.shape[dims[0]])
                    condition[distances.argmin(axis=dims[0])]=1
                    conditions=(condition,)
                            
                elif rule=='vicinity_avg':
                    desired_val,tolerance=desired_val
                    distances=abs(expr_val-np.ones(expr_val.shape)*desired_val)
                    nd_condition=(distances/desired_val)<tolerance
                    if len(dims)>1:
                        conditions=[nd_condition.any(axis=list(dims).remove(dim)) for dim in dims]  # converge all dims but the one for which the rule is
                    else:                        # in case the rule is for only one dim, all but one means nothing. 
                        conditions=(nd_condition.flat,)  # However, if axis=(), numpy converges every dimension, so this is a workaround 
                        
                elif rule=='range_avg' or rule=='range_sum':
                    desired_range_start,desired_range_end=sorted(desired_val)
                    #print desired_range_start,desired_range_end #DEBUG
                    nd_condition=np.logical_and(desired_range_start<expr_val,expr_val<desired_range_end)
                    if len(dims)>1:
                        conditions=[nd_condition.any(axis=list(dims).remove(dim)) for dim in dims]  # converge all dims but the one for which the rule is
                    else:                        # in case the rule is for only one dim, all but one means nothing. 
                        conditions=(nd_condition.flat,)  # However, if axis=(), numpy converges every dimension, so this is a workaround 
                            
                for i,var in enumerate(var_set):
                    for condition,dim in zip(conditions,dims):
                        if var.shape[dim]>1:
                            var_set[i]=var.compress(condition, axis=dim)
                        if rule=='range_sum':
                            var_set[i]=var_set[i].sum(axis=dim)
                        else:
                            var_set[i]=var_set[i].mean(axis=dim)
                        
                for i,var in enumerate(eval_vars):
                    for condition,dim in zip(conditions,dims):
                        if var.shape[dim]>1:
                            eval_vars[i]=var.compress(condition, axis=dim)
                        if rule=='range_sum':
                            var_set[i]=var_set[i].sum(axis=dim)
                        else:
                            var_set[i]=var_set[i].mean(axis=dim)
                    
         
        #convert everything to desired units
        
        for i,var in enumerate(var_set):
            if var_units[i] is not None:
                var_set[i]=var.rescale(var_units[i])
    
        # contract on unused dimensions
        unused_dims=[]
        for i in xrange(len(var_set[0].shape)):
            if (np.array([var.shape[i] for var in var_set])==1).all():
                unused_dims.append(i)
        for dim in unused_dims:
            for i,var in enumerate(var_set):
                var_set[i]=var.squeeze(axis=dim)
                
        # normalize - last variable in var_set by default
                
        normval = keywords.pop('normalize',False)
        if normval:
            normval = 1.0 if normval==True else np.float(normval)
            var_set[-1]=var_set[-1]/np.max(var_set[-1])*normval
             
                
                    
        # retranspose arrays to arrange dimensions according to input set
        #list_of_dims=[]
        #new_var_set=[]
        #for index,var in enumerate(var_set):
        #    list_of_dims.append(var.ndim-1)
        #    new_var_set.append(pq.Quantity(
        #                                        np.array(var.transpose(),ndmin=index+1,copy=False)
        #                                   ).transpose())
        #if len(list_of_dims)<indep_var.ndim:
        #    raise ValueError('Underparametrized expressions. Dependent variable depends on more axis than provided')
        #var_set.append(indep_var.transpose(list_of_dims))
        return var_set, keywords  


# this is a convenience function as keywords are usually not needed anywhere,
# except in plotting functions
    def get_table(self,*args,**keywords):
        return self._get_table(*args,**keywords)[0]  
             
########################################################################
    def get_table2d(self,*args,**keywords):
        '''\
Old behaviour of get_table: only 1d peTabs are supported.
Keywords: 
    X,Y -- the expressions for x and y
    
    filters --  a tuple of definitions of filters to be applied to the data 
        filter definition is a dictionaty with one element: 
            its key is a string of filter type 
                for example 'adjavg' , 'fourier' , 'golay' or 'cosmic'
            its value is a filter parameter or a tuple of filter parameters (if several required)
    
'''
    # load all 'our' parameters               
        xrange=keywords.pop('xrange',[-np.inf,+np.inf])
        Xexpr=keywords.pop('X',self['defX'])
        Yexpr=keywords.pop('Y',self['defY'])      
        normalize_y=keywords.pop('normalize',0)
        filters=keywords.pop('filters',None) # 'adjavg' , 'fourier' , 'golay' or 'cosmic'
    # evaluate XY
        Xcols=self.eval_expr(Xexpr)
        Ycols=self.eval_expr(Yexpr) 
        if not isinstance(Xcols, (tuple,list)):
            Xcols=[Xcols,]
        if not isinstance(Ycols, (tuple,list)):
            Ycols=[Ycols,]
            
        dim=max(len(Xcols),len(Ycols))
        if dim !=1:
            if len(Xcols)==1:
                Xcols=Xcols*dim
            if len(Ycols)==1:
                Ycols=Ycols*dim
        if len(Xcols)!=len(Ycols):
            raise ValueError('X and Y expression sets do not evaluate to equal amount of datasets')
            
        XYpairset=[]    
        for Xcol,Ycol in zip(Xcols,Ycols):  
            #build x,y pairs for sorting
            units=(Xcol.units,Ycol.units)
            XYpairs=zip(Xcol,Ycol)
            # remove NaN's due to bug in python (max(nan,3,4))   
            newXYpairs=[]
            for x,y in XYpairs:
                if (not math.isnan(x)) and ((not math.isnan(y))):
                    newXYpairs.append((x,y))
            XYpairs=newXYpairs

            # sort data with ascending X
            XYpairs=sorted(XYpairs,key=lambda x: x[0])
            
            #convert back to columns      
            Xcol,Ycol=(np.array(XYpairs)[:,0]*units[0],np.array(XYpairs)[:,1]*units[1])
              
            # apply filtering if requested   
            if(filters):
                Ycol=filter_wrapper(Ycol,filters) 
            if normalize_y :  # divide by maximum
                Ycol=Ycol/max(Ycol)*normalize_y
            XYpairset.append([Xcol,Ycol])     
       
        return XYpairset
#########################################################################################
    def _plot2d_single_y(self,Xcol,Ycol,*args,**keywords):
        '''\
A simple plotting function for a single Y variable 
'''
                                
        #set default plot parameters
        if 'plotjoined' not in keywords:
            keywords['plotjoined']=True;
        if 'frame' not in keywords:
            keywords['frame']=True;    
        if 'axes' not in keywords:
            keywords['axes']=False;
        if 'axes_pad' not in keywords:
            keywords['axes_pad']=0;  
        if 'axes_labels_size' not in keywords:
            keywords['axes_labels_size']=1; 
       

                     
        #plot!
        from sage.plot.plot import list_plot
        # list_plot does not accept any args - it only treat the second arg as a plotjoined value
        # so we skip passing *args into..
        return list_plot(zip(Xcol,Ycol),**keywords)
#########################################################################################
       
    def plot2d(self,*args,**keywords):
        '''\
A simple plotting function like an old one

Keywords:

All keywords of sage.all.list_plot() are supported. 
The following oned have re-defined defaults:

'plotjoined' 
    is true by default, but can be set to false to plot dotted graph
'frame' 
    is true by default, controls box around the plot
'axes'
    set to true to plot crossed axes at (0,0)
'axes_pad'
    default axes padding is zero
'color'
    if not set and multiple graphs to be plotted, color is cycling BRGCMYK
'axes_labels' 
    a function tries to guess those from the X and Y expression names. 
    Can be set explicitly.

The following keywords are specific to sage_plot():

'X' 
    the expression to plot along X axis. default is defX
'Y'
    the expression to plot along Y axis. default is defY

'logscale'
    a tuple with 2 booleans. defaults to (False,False). 
    deprecated since sage has its own log plots
    use scale='semilogy' and scale='loglog'
    it is possible to set log base, ex: base=2

'filter_method'
    can be 'adjavg' , 'fourier' , 'golay' or 'cosmic'. 
    Applies adjacent averaging, Fourier lowpass, Savitsky-Golay or cosmic ray filtering.
    any other value does nothing - as is by default

'filter_param'
    a parameter(s) for filter to be applied

    Adjacent averaging and Fourier: a single integer ``min_period`` -- a threshold for the lowpass filter
        (the minimal period of the oscillations which should be left intact) expressed in a number of 
              samples per one full oscillation.

    Savitsky-Golay: a tuple of integers (pol_degree,diff_order)
           pol_degree is degree of fitting polynomial

        diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first derivative of function.
                     and so on ...)
    Cosmic ray removal: a single float ``tolerance`` 
        Filter searches for separate outliers in data, which are away 
        from both neighbours more than tolerance*(average difference between neighboring points),
        and sets them to average of neighbors.
        The bigger is the tolerance, the less data are affected.
        Makes no sence to make tolerance lower than  1.0

'textlegend' 
    if true (default) prints a text legend based on graph colors to stdout 
    
'textlegend_prec'
    a precision of values in textlegend. Defaults to 3 digits after decimal point.

'''     
        # load all 'get_table' parameters               
        #xrange=keywords.pop('xrange',[-np.inf,+np.inf])
        #Xexpr=keywords.pop('X',self['defX'])
        #Yexpr=keywords.pop('Y',self['defY'])      
        #normalize_y=keywords.pop('normalize',0)
        #filters=keywords.pop('filters',None) # 'adjavg' , 'fourier' , 'golay' , 'cosmic' , 'FIR'
        
        # load expr_list to know what we are plotting
        expr_list, keywords = extract_multivar_args(
            args, keywords,
            single_keywords=('X','Y','Z'),
            single_kwd_defaults=(self['defX'],self['defY'],self['defZ']),
            compound_keyword='expr_set')
  
         
        Xexpr=expr_list[0]
        Yexprs=expr_list[1:] 
           
        # extract 'our' arguments
        textlegend=keywords.pop('textlegend',True)
        textlegendprec= keywords.pop('textlegend_prec',3)
        
        removeNan= keywords.pop('remove_Nan',True)
        # create textlegend
        text_legend_data=self.short_pars(names=False,prec=textlegendprec)

        #concatenate Xs and Ys 
        #get a table 
        # the following extracts get_table()-related keywords from **keywords
        # this is required because sage's plotting functions yell too much
        # if they are passed a foreign keyword argument
        XYYYtable, keywords = self._get_table(expr_set=expr_list,**keywords)
        # to array
        XYYYtable=np.array(XYYYtable)
        
        # remove non-plottable values, as list_plot does not handle them gracefully 
        if (removeNan):
            XYYYtable=XYYYtable[:,~np.isnan(XYYYtable).any(0)]
        
        #try:
        #    if (keywords['scale'] in ('semilogy','loglog')):   
        #        XYYYtable=XYYYtable[:,(XYYYtable[1:]>0).all(0)] 
            
        #    if (keywords['scale'] in ('semilogx','loglog')):   
        #        XYYYtable=XYYYtable[:,(XYYYtable[0]>0)]    
        #except KeyError:
        #    pass  
        
        
        # back to list of columns    
        #XYYYtable=np.vsplit(XYYYtable,XYYYtable.shape[0])
            
        Xcol=XYYYtable[0]
        YYtable=XYYYtable[1:]
               
        
        #extract units for variables
        allunits=[]
        for col in XYYYtable:            
            try: 
                units=col.units
            except AttributeError:
                units='arb. units'        
            if (units==pq.dimensionless):
                units='arb. units'
            else:
                units=pretty_str_units(units)
            allunits.append(units)               
        yunits=allunits[1:]
        #construct labels;   replace underscores with spaces 
        unit_strings=[repr(Yexpr)+', '+str(units) for Yexpr,units in zip (Yexprs,yunits)]
           
        from re import sub
        xlabel=sub('_',' ',repr(Xexpr)+', '+str(allunits[0]) )
        ylabel=sub('_',' ',';'.join(unit_strings) )
        
        keywords['axes_labels']=keywords.pop('axes_labels',[xlabel,ylabel])                        
                                
        # TODO: Clear all foreign kwds...                        
        # set default color                         
        if  'color' in keywords:
            def colorspec():
                return keywords.pop('color')
        else:                       
            coloriter=ColorBank() 
            def colorspec():
                return coloriter.next()
                
        from sage.plot.graphics import Graphics
        graph=Graphics()                                       
        for Ycol in YYtable:
            nextcolor=colorspec()
            text_legend_data+= ' - ' + repr(nextcolor)
            graph+=self._plot2d_single_y(Xcol,Ycol,*args,
                                        color=nextcolor, **keywords)    
          
        if (textlegend):          
            print text_legend_data     
        return graph   
        
         
################################################################################
##  Representation-related routines:
##  pretty-printing, showing and stuff
################################################################################
            
    def short_pars(self,names=True,prec=3):
        '''\
Show the values of parameters specified by strings in self['_params_to_show']
'''     
        # this was an attempt to set the precision of output.
        # however, quantities package does not implement this. 
        
        #oldprec=np.get_printoptions()['precision']
        #np.set_printoptions(precision=3)
        
        # using np.around() instead
        output=''
        for par in self['_params_to_show']:
            outval=self.eval_expr(var(par))   
            try:
                outval=np.around(outval,prec)
            except TypeError:
                pass
            output+=(par+':' if names else '') + str(outval) + '    '
                           
        #np.set_printoptions(precision=oldprec)
        return output
                        
    def pretty_print(self):
        print self.pretty_str()
                        
    def pretty_str(self):
        
        def sorting_func(item):
            # first are the _params_to_show
            parlist =self['_params_to_show']
            if repr(item[0]) in parlist:
                return parlist.index(repr(item[0]))
            maxidx=len (parlist)   
            if isinstance(item[1],(np.ndarray,pq.Quantity)):
                if item[1].shape==() or item[1].shape==(1,):
                    return maxidx+1 # 0-D parameters go second
                else:
                    return maxidx+2 # tabulated parameters go last
            elif not isinstance(item[1],sage_expr):
                return maxidx+0 # strings, dates and other go first
            else:
                return maxidx+3 # expression-parameters go last
       
        outp=''
        #
        sorteditems=sorted(self.items(),key=lambda x:repr(x[0]))
        sorteditems.sort(key=sorting_func)
        for par,value in sorteditems:
            outp+=par + ': '
            if isinstance(value,(np.ndarray,pq.Quantity)):
                if value.shape==() or value.shape==(1,):
                    outp+=str(value)
                else:
                    if hasattr(value,'units'):
                        units_str=pretty_str_units(value.units)
                    else :
                        units_str='dimensionless'
                    outp+=repr(value.ndim) + '-D array, ' + units_str
            elif isinstance(value,datetime.datetime):
                outp+=str(value)
            else: # if isinstance(value,(np.ndarray,pq.Quantity))
                outp+=repr(value)
            outp+='\n'
        del (sorteditems)
        return outp
        
        
                            
                        
