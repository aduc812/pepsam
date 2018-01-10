from sage.symbolic.ring import SR
from sage.symbolic.expression import Expression as sage_expr
from .utils import _expression_to_dict, _expression_from_dict
from  numpy import ndarray


ESET_KEY='expression_set'
    
def E(*args):
    '''E is a short alias to Expr_set((arg1,arg2,arg3...))

Expr_set or E is a tuple of expression objects, 
with redefined numeric operations so that a+b does not mean concatenation
but rather an element-wise sum of two Expr_set objects, like in numpy arrays.       
'''    
    if len(args)==1:
        if isinstance(args[0],(Expr_set,list,tuple,ndarray)):
            return Expr_set(args[0])
        #else:
        #    return SR(args[0])
    return Expr_set(args)
    
def _expr_set_from_dict(expr_set_dict):
    return Expr_set([_expression_from_dict(dict_) for dict_ in expr_set_dict[ESET_KEY]])  


###############################################################################
# Sage-specific expression functions - automated adding as a bunch
# using function and class decorators
# this does not work in this state and considered not useful at this stage
###############################################################################
#from  functools import wraps
#def _element_wise(function): 
#    print 'in element-wise:' + repr(function) 
#    @wraps(function)  
#    def wrapper(self,*args, **kwargs):
#       print 'executing decorated func; self='+ repr(self) + ' args=' + repr(args) 
#       return pm.Expr_set([function(i,*args,**kwargs) for i in self._explist])
#    print new_function
#    return new_function

#def _add_decorated_methods_from_sage_expr(decorator):
#    def _apply_wrapped_methods_from_sage_expr(cls):
#        print cls
#        for attr,attr_def in sage_expr.__dict__.items():
#            if callable(attr_def) and attr[0:2]!='__':
#                if attr not in cls.__dict__:
#                    setattr(cls, attr, decorator(attr_def))
#        return cls
#    return _apply_wrapped_methods_from_sage_expr
    
###############################################################################
# Define Expr_set class
# 
###############################################################################   
from sage.structure.sage_object import SageObject 
#@_add_decorated_methods_from_sage_expr(_element_wise)
class Expr_set (SageObject):#(sage_expr): #

    def __init__(self,iterable=[]):
        self._explist=tuple([SR(i) for i in iterable])
# Emulating container type
        
    def __iter__(self):
        return self._explist.__iter__()

    def __reverse__(self):
        return self._explist.__reverse__()

    def __len__(self):
        return self._explist.__len__()
        
    def __getitem__(self,index):
        return self._explist.__getitem__(index)
        
    def __contains__(self,item):
        return self._explist.__contains__(item)

#    item removal routines - not natural for tuple-based Expr_set
        
#    def removed(self,item):
#        return Expr_set(list(self._explist).remove(item))
#        
#    def all_removed(self,item):
#        new_explist = list(self._explist)
#        while(True):
#            try:
#                new_explist.remove(item)
#            except ValueError:
#                break
#        return Expr_set(new_explist)
        
# define sage-style _repr_    
    def _repr_(self):
        return 'Expr_set('+repr(self._explist)+')'
# as well as python __repr__       
    def __repr__(self):
        return self._repr_()
          
    def __str__(self):
        return str(self._explist)
        
    def _latex_(self):
        from sage.misc.latex import latex
        return latex(self._explist)
###############################################################################
# Unificated binary math operations
###############################################################################
    def _single_binary_math(self,other,function):
        return Expr_set([function(i,SR(other)) for i in self._explist])
    def _iterable_binary_math(self,other,function):
        return Expr_set(map(function,self._explist,other))
    def _iterabletosingle_binary_math(self,other,function):
        return Expr_set([function(self._explist[0],SR(i)) for i in other])    
        
    def _binary_math(self,other,function):
        if isinstance(other,basestring):
            raise ValueError('Cannot coerce strig to Expression')       
        try:
            iter(other)
        except TypeError:
            return self._single_binary_math(other,function)
        else:
            if len(other)==1:
                return self._single_binary_math(other[0],function)
            if len(self._explist)==1:
                return  self._iterabletosingle_binary_math(other,function)
            if len(self._explist)==len(other):
                return self._iterable_binary_math(other,function)
            raise ValueError('Cannot perform a binary operation on iterables of different length')
###############################################################################
# end of Unificated binary math operations
###############################################################################  

###############################################################################
# Regular math operations
###############################################################################   
    def __add__(self,other):
        return self._binary_math(other,sage_expr.__add__)
            
    def __radd__(self,other):
        return self.__add__(other)
        
    def __neg__(self):
        return Expr_set(map(sage_expr.__neg__,self._explist))
        
    def __pos__(self):
        return Expr_set(self._explist)
        
    def __abs__(self):
        return Expr_set(map(abs,self._explist))
        
    def __invert__(self):
        return Expr_set(map(sage_expr.__invert__,self._explist))
        
    def __rsub__(self,other):
        return (self.__neg__()).__add__(other)
    
    def __sub__(self,other):
        return (self.__rsub__(other)).__neg__()
        
    def __mul__(self,other):
        return self._binary_math(other,sage_expr.__mul__)
            
    def __rmul__(self,other):
        return self.__mul__(other)
            
    def __rdiv__(self,other):
        return (self.__invert__()).__mul__(other)
        
    def __div__(self,other):
        return (self.__rdiv__(other)).__invert__()
    
    def __truediv__(self,other):
        return self.__div__(other)
    
    def __rtruediv__(self,other):
        return self.__rdiv__(other)
        
    def __pow__(self,other):  
        return self._binary_math(other,sage_expr.__pow__)
            
###############################################################################
# dumping to dictionary - for storing in mongodb
############################################################################### 
            
    def dict(self):
        return {ESET_KEY:[_expression_to_dict(expr) for expr in self]}
        
###############################################################################
# __eq__ - TODO: define also __hash__
###############################################################################        
        
    def __eq__(self,other):
        if not isinstance(other, Expr_set):
            return False
        return self._explist.__eq__(other._explist)
            
    def arguments(self):
        all_args=[]
        for expr in self._explist:
            for arg in expr.arguments():
                if arg not in all_args:
                    all_args.append(arg)
        return tuple(all_args)
              
    def __call__(self,**keywords):
        if keywords=={}:
            return self
        kwdlengths=[]
        for var,val in keywords.items():
            try:
                iter(val)
            except TypeError:
                kwdlengths.append(0)
            else:
                kwdlengths.append(len(val))
                
        kwdlen=max(kwdlengths)
        if kwdlen<1:
            kwdlen=1        
        for i in kwdlengths:
            if i not in (0,1,kwdlen):
                raise ValueError('got keyword iterables of different length')       
        if (len(self._explist) not in (1,kwdlen) ) and kwdlen != 1 :
            raise ValueError('length of keyword iterables does not match \
the length of expression set')
        if len(self._explist)>1 and   kwdlen==1:
            kwdlen=len(self._explist)
        mapped_kwds_var=[]
        mapped_kwds_val=[]   
        for klen,(var,val) in zip(kwdlengths,keywords.items()):

            if klen==kwdlen:
                mapped_kwds_var.append(var)
                mapped_kwds_val.append(val)

            elif klen==0:
            # here below multiplication of tuple means concatenation:
                mapped_kwds_var.append(var)
                mapped_kwds_val.append(Expr_set((val,)*kwdlen))
            else: # klen=1
            # here below multiplication of tuple means concatenation:
                mapped_kwds_var.append(var)
                mapped_kwds_val.append(Expr_set((val[0],)*kwdlen))

        srt_kwds=[]#[[{k:v} for k,v in zip(mapped_kwds_var,i)] for i in zip(*mapped_kwds_val)]
        for i in zip(*mapped_kwds_val):
            locdct={}
            for k,v in zip(mapped_kwds_var,i):
                locdct.update({k:v})
            srt_kwds.append(locdct)    
             
        if len(self._explist)>1:
            return Expr_set([i.__call__(**pars) for i,pars in zip(self._explist,srt_kwds)])
        else:
            return Expr_set([self._explist[0].__call__(**pars) for pars in srt_kwds])             
    
#    def _sinlecall(self,**keywords):
#        return Expr_set([expr.__call__(**keywords) for expr in self._explist])
 
 

              
        
        
