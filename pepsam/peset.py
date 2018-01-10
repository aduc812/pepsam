
import quantities as pq
import re, mmap

re_datablock=re.compile(r'((^(([-\d.+e]+[\t ]*)+[\r\n]+))+)',flags=re.MULTILINE)

class pePar(object):
    def __init__(self,name,**kwds):
        self.name=name
        self.units=kwds.pop('units',None)
        self.type=kwds.pop('type','float')
        self.default=kwds.pop('default',None)
        self.printorder=kwds.pop('printorder',None)
        
        # some consistency checks
        if self.type=='quantity' and self.units==None:
            self.units=pq.dimensionless            
        if isinstance (self.units, pq.quantity.Quantity):
            self.type='quantity'
        else:
            if self.units is not None:
                raise TypeError('units should be of \'pq.quantity.Quantity\' type')

class peSet(object):
    '''This class is a base class for physical experiment setup,
     which creates physical experiment Tables'''
     
    def __init__(self):
        self.parameters=[
                            pePar('experiment_datetime',type='datetime'),
                            pePar('setup_name',type='string'),
                        ]
        self.file_regexps=[re_datablock,]
        

        
    def parse_file(self,filename):
        with open(filename, 'r+') as f:
            data = mmap.mmap(f.fileno(), 0)

            mo = re_datablock.search(data)
            if mo:
                print str(mo.groups()[0])
                        
