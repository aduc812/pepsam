## numpy
import numpy as np

## physical quantities
import quantities as pq

##sage imports
#import sage



## other tools
import re
import warnings
from bson import InvalidDocument

def dbWaveformSet(dpkg):
    '''
    save to db experiments stored as values in dictionary. 
    The fuction prints a string which, when evaluated, 
loads all experiments into same dictionary 
'''
    dbkeys=dict()
    for d in dpkg:
        if dpkg[d] is None:
            print None
            dbkeys.update({d:None})
        else:
            try:
                dbkeys.update({d:dpkg[d].db()})
            except RuntimeError as e:
                print str(d)+ '\t' + str(e)
                dbkeys.update({d:None})
            except (InvalidDocument, AttributeError):
                dbkeys.update({d:None})
    #print dbkeys
    print 'dpkg={\n'
    for key,value in dbkeys.items():
        print str(key)+':peTab("'+str(value)+'"),'
    print '}'
    return dbkeys
    
dbExpList=lambda dpkg: dbWaveformSet(dpkg)   
'''
    An alias to dbWaveformSet()
'''

def expgraph(gr,preview=True,filename='graph0',file_format='pdf',**kwargs):
    '''
    Shows the graph in the worksheet and if preview = False
    exports it into pdf with specified filename.
'''
    gr.show(**kwargs)
    if file_format in ['pdf','svg','png']:
        extenstion='.'+file_format
    else:
        extenstion='.svg'
    if not preview:
        gr.save(filename+extenstion,**kwargs)
        
        # postprocess - reqiures inkscape installed
        if not (file_format in ['pdf','svg','png']):
            import subprocess 
            retval = subprocess.call(["which", "inkscape"])
            if retval != 0:
                RuntimeError("inkscape not installed - cannot convert from svg")
        
            if file_format=='eps':
                from subprocess import call
                call(["inkscape", filename+extenstion,'-E', filename+".eps"])
                call(["rm", filename+extenstion]) # cleanup
            else:
                RuntimeError("unsupported output format")
    


def tableout(tuple_of_XYlist_pairs):
    '''
    Concatenates a few x-y column pairs to a single x-y column pair and saves it into a 2-column data file. Does not sort anything.
'''
    Xs=np.array([])
    Ys=np.array([])
    for XYlist_pair in tuple_of_XYlist_pairs:
        Xs=np.concatenate((Xs,XYlist_pair[0]))
        Ys=np.concatenate((Ys,XYlist_pair[1]))
    return Xs,Ys

def readXY(filename,sep=None,comment_chars='#',cols=(0,1),decimal_sep='.'):
    '''
    Loads multicolumn data file into a list of arrays, each array corresponds to the column in file. 
    filename - the name of the file to load
    sep - column separator. None corresponds to any number of whitespace chars, see also str.split().
    comment_chars - a string or a list of chars which mark comment lines in file
    cols - a list of column numbers to load
'''
 
    xyypoints=[]

    with open(filename,'rU') as f:
        for line in f:
            if line[0] in comment_chars:
                continue
            try:                  
                xyyrow=[line.split(sep)[col] for col in cols]
            except (ValueError,IndexError) as e:
                warnings.warn(str(e)+', ignoring line(s)', RuntimeWarning)
                continue    
            try: 
                xyypoints.append( [float(val.replace(decimal_sep,'.')) for val in xyyrow] )
            except ValueError as e:
                warnings.warn(str(e)+', ignoring line(s)', RuntimeWarning)
                continue    

    return [np.array(col) for col in zip (*xyypoints)]


def IntegerBank():
    '''\
A simple integer generator.
Usage:
    sage:seq=IntegerBank()
    sage:seq.next()
    0
    sage:seq.next()
    1
    
'''
    i=0
    while(1):
        yield i
        i+=1

def ColorBank():
    '''\
A simple color generator.
Usage:
    sage:color=ColorBank()
    sage:color.next()
    'blue'
    sage:color.next()
    'red'
    
'''
    while(1):
        yield 'blue'
        yield 'red'
        yield 'green'
        yield 'cyan'
        yield 'magenta'
        yield 'orange'
        yield 'black'
        yield 'yellow'
        yield 'purple'
        yield 'grey'


unitdict={}
'''\
unitdict is a dictionary of known units
contains mapping of unit name strings
to corresponding quantities.UnitQuantity instances
    
    'eV' => pq.eV
'''
for item,value in vars(pq).items():
    if isinstance(value,pq.UnitQuantity):
        unitdict.update({item:value})
        
units_and_array=unitdict.copy()
'''\
units_and_array is a dictionary containing mapping of 
unit names to corresponding quantities.UnitQuantity instances
    
    'eV' => pq.eV
and 'array' => numpy.array in addition.

Is used internaly to generate context for reading units from string representation 
'''    
units_and_array.update({'array':np.array})

# compile regular expressions to make em faster 
re_array1=re.compile('array\(1\.0\) \* ')
re_1=re.compile('1\.0 ')

def pretty_repr_units(units):
    return re_array1.sub('',repr(units))
    
def pretty_str_units(units):
    return re_1.sub('',str(units))
    

def _quantity_to_dict(pqval):
    return {
            'value':pqval.magnitude.tolist(),
            'units':pretty_repr_units(pqval.units)
            }
def _quantity_from_dict(pqdict):
    from sage.misc.sage_eval import sage_eval
    return (
            np.array(pqdict['value']) *
            sage_eval(pqdict['units'], locals=units_and_array,preparse=False)
            )



def _expression_to_dict(expr):
    from sage.symbolic.expression import Expression
    if not isinstance(expr,Expression):
        raise ValueError('expected expression object, got  '+repr(expr))
    return {
                'expression':repr(expr),
                'arguments':[repr(arg) for arg in expr.arguments()],
           }
           
           
def _expression_from_dict(expr): 
    from sage.misc.sage_eval import sage_eval
    from sage.calculus.calculus import var          
    # assume we have checked the dictionary already
    return sage_eval(
                expr['expression'],
                locals=dict([(
                                arg,
                                eval(   'var("' + arg + '")'  )
                              ) for arg in expr['arguments']])
                     )

def dev_dims(array_like):
    '''
    Checks the array shape and returns the indeces of "developed"
        dimensions, i.e. the length of array along those greater
        than unity 
'''
    dims=[]
    for i,dim in enumerate(array_like.shape):
        if dim>1:
            dims.append(i)
    return dims   
    
    
def extract_multivar_args(args=[],kwds={},
                            single_keywords=[],
                            single_kwd_defaults=[],
                            compound_keyword=None,
                            ):
    '''\
    The function pases args and certain kwds to handle multivariant 
    input of a caller function. The result is a ordered sequence of arguments,
    which can be passed in a various ways (in the order of priority):
    
    1) if  compound_keyword is in kwds, then its value is returned 
        as the desired list
    2) if there is a single argument (args[0]), it is converted into 
        a list and returned. If it is not an iterable, it is wrapped 
        into a list        
    3) if there are multiple arguments, the list of arguments is returned
    4) if all the above fails, the list is constructed from the values of 
        single keyword arguments provided in single_keywords, and in case 
        they are missing, the values from single_kwd_defaults are used.
    
    Output: (desired_values_list, clean_kwds)
        Where clean_kwds is a dictionary of all remaining keywords in kwds
    
'''
    skw_values=[]
    for kwd,deflt in zip(single_keywords,single_kwd_defaults):
         skw_values.append(kwds.pop(kwd,deflt) )
         
    if len(args)==0: # no args, single  notation used
            #skw_values=skw_values
            pass
    else: 
        if len(args)==1: # single argument, treat as list 
            try:
                skw_values=list(args[0])
            except TypeError:  # single expression, create list
                skw_values=list(args,)
        else: # multiple expressions, create list
            skw_values=list(args)
    if compound_keyword is not None:
        skw_values=kwds.pop(compound_keyword,skw_values)
    return skw_values, kwds

import math
#import pyaudio
import sys

def beep_once():
    '''this function only makes sense if run on localhost
As this is not an intended use of the package anymore, 
It is commented out'''
    pass
#    PyAudio = pyaudio.PyAudio
#    RATE = 16000
#    WAVE = 1000
#    data = ''.join([chr(int(math.sin(x/((RATE/WAVE)/math.pi))*127+128)) for x in xrange(RATE/5)])
#    p = PyAudio()
#
#    stream = p.open(format =
#                p.get_format_from_width(1),
#                channels = 1,
#                rate = RATE,
#                output = True)
#    for DISCARD in xrange(5):
#        stream.write(data)
#    stream.stop_stream()
#   stream.close()
#    p.terminate()
    
def test_ipython():
     
    try:
        get_ipython()
    except NameError:
        return False
    else:
        return True
