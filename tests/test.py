# a sample test file

from sage.all import *
from tempfile import mkdtemp
import os
from shutil import rmtree

from .context import pemongo # this way to import the main package

#old_stdout = sys.stdout

#from StringIO import StringIO
#out=StringIO()

#sys.stdout=out
#path_to_dir=os.getcwd()
#try:
print 'hello world'
plot(sin).save('fig.svg')
    
#finally:
#    sys.stdout = old_stdout                    
