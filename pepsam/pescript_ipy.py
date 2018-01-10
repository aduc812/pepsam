# -*- coding: utf-8 -*-
# we assume IPython is running otherwise this module never gets loaded
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic)
from .pescript import peScript                                
from .utils import test_ipython
                                
@magics_class
class peScriptMagic(Magics):
    '''
This is a helper magics which allows to create peScripts from jupyter cells. See pemongo.peScript
usage: %%peScriptm  (description,oid,keywords)
the "arguments" are taken as string and then evaluated. There should be exactly three of them.
    '''

    #def __init__(self, shell, data):
        # You must call the parent constructor
    #   super(StatefulMagics, self).__init__(shell)
    #    self.data = data
    #    print (data)
        
    @cell_magic
    def peScriptm(self, args, code): #desc=None,oid=None,keywords=None,code=''):
        args=eval(args)
        script=peScript(desc=args[0],oid=args[1],keywords=args[2],code=code)
        script.eval(code, globals())
        # show images in script
        


# This class must then be registered with a manually created instance,
# since its constructor has different arguments from the default:
if (test_ipython()):
    try:
        ip = get_ipython()
        ip.register_magics(peScriptMagic)
        #magics = StatefulMagics(ip, '')
        #ip.register_magics(magics)
    except NameError:
        pass
