# encoding: utf-8
from .context import pemongo
from pemongo.petab import peTab
from pemongo.expr_set import E
from pemongo.filters import filter_wrapper
from pemongo.mongo_connection import mongoTable
from pemongo.expr_set import Expr_set

from bson import ObjectId
import datetime as dt
import pytz
import numpy as np
import quantities as pq
import unittest
import random
from sage.calculus.calculus import var
import sage


randomUnit=lambda : pemongo.petab.unitdict[random.choice(pemongo.petab.unitdict.keys())]


class TestComparison(unittest.TestCase):
    def setUp(self):
        random.seed()
        self.maxlen=100000
        maxlen=self.maxlen
        self.p={
                    'randomlist':[random.random() for i in range(0,random.randint(0,maxlen))],
                    'randomfloat':random.random(),
                    'randomarray':np.array([
                                            random.random() 
                                            for i in range(0,random.randint(0,maxlen))
                                            ]),
                    'randomQuantity':np.array([
                                                    random.random() 
                                                    for i in range(
                                                                   0,
                                                                   random.randint(0,maxlen)
                                                                   )
                                            ])*randomUnit(),
                   'wavelength':1239.8*var('eV')*var('nm')/var('energy'),
                   'complex1':var('wavelength')/0.0000138*var('intensity')**2,
                   'complex2':var('complex1')*var('energy')/var('wavelength')**5,
                   'expset':E(var('complex1'),var('complex2'),var('wavelength')),  
               }
        self.q={
                    'randomfloat':random.random(),
                    'randomarray':np.array([random.random() for i in range(0,random.randint(0,maxlen))]),
                    'randomQuantity':np.array([random.random() for i in range(0,random.randint(0,maxlen))])*randomUnit(),
                    'wavelength':1239.8*var('eV')*var('nm')/var('energy'),
                    'complex1':var('wavelength')/0.0000138*var('intensity')**2,
                    'complex2':var('complex1')*var('energy')/var('wavelength')**5, 
                    'expset':E(var('complex1'),var('complex2'),var('wavelength')),     
               }
               
    def test_copy(self):
        maxlen=self.maxlen
        pp=peTab(self.p)           
        qq=pp.copy()
        randomlist_idx=random.randint(0,len(qq['randomlist']))
        qq['randomlist'][randomlist_idx]+=random.randint(1,maxlen)
        self.assertNotEqual(qq['randomlist'][randomlist_idx],pp['randomlist'][randomlist_idx])
 
        rq_idx=random.randint(0,len(qq['randomQuantity']))
        qq['randomQuantity'][rq_idx]+=random.randint(1,maxlen)*qq['randomQuantity'].units
        self.assertNotEqual(qq['randomQuantity'][rq_idx],pp['randomQuantity'][rq_idx])
              
    def test_of__eq__method(self):
        pp=peTab(self.p)
        qq=peTab(self.q)
        self.assertTrue(pp==pp)
        self.assertEqual(pp,pp.copy())
        self.assertEqual(qq,qq.copy())
        self.assertFalse(pp!=pp.copy())
        self.assertNotEqual(pp,dict(pp))
        self.assertNotEqual(pp,qq)
        ppm=pp.copy()
        ppm.update({'trololo':'kaboom'})
        self.assertNotEqual(pp,ppm)
       

class TestCreationSavingLoading(unittest.TestCase):

    def setUp(self):
        random.seed()
        self.a=np.array(range(1,130))
        self.b=np.array((1,2,3,4,5),ndmin=2).transpose()
        self.c=np.array((1,0.5,2),ndmin=3).transpose()
        self.ab=(np.ones(self.b.shape)*self.a)*self.b
        self.abc=(np.ones(self.c.shape)*self.ab)*self.c
        self.intt=(np.sin(np.exp(self.a))+self.a/50.0)
        self.intt[67]=40
        self.p={
            'datetime':dt.datetime(2012, 12, 25, 20, 35, 23, 0,pytz.UTC),
            'random_value':{'value':random.random()},
            'sample':'my_favourite_sample',
            'unicodestr':u'раз два 草 直',
            'laser_power':{'value':150,'units':'W'},
            '_params_to_show':['sample','laser_power'],
            'energy':{'value':self.a.tolist(),'units':'eV*kg'},
            'concentration':{'value':self.c.tolist(),'units':'1/mol'},         
            'temperature':{'value':self.b.tolist(),'units':'K'},
            'intensity':{'value':self.intt.tolist(),'units':'dimensionless'},
            'wavelength':1239.8*var('eV')*var('nm')/var('energy'),
            'complex1':var('wavelength')/0.0000138*var('intensity')**2,
            'complex2':var('complex1')*var('energy')/var('wavelength')**5,
            'expset':E(var('complex1'),var('complex2'),var('wavelength')),   
        }
        self.created_db_record=None

    def test_creation(self):
        tab=peTab(self.p)
        # check some values         
        self.assertEqual(tab['laser_power'],150.0*pq.W)
        self.assertEqual(tab['datetime'],dt.datetime(2012, 12, 25, 20, 35, 23, 0,pytz.UTC))
        self.assertTrue(isinstance(tab['complex1'],sage.symbolic.expression.Expression))
        # check array shapes
        self.assertEqual(tab['laser_power'].shape,())
        self.assertEqual(tab['temperature'].shape,(5, 1))
        self.assertEqual(tab['energy'].shape,(129,))
        self.assertEqual(tab['concentration'].shape,(3, 1, 1))
        # check all params exist
        for par in self.p:
              self.assertTrue(par in tab)
        
    def test_saving_to_dict_and_loading(self):
        tab=peTab(self.p)
        dicttab=tab.dict()
        copytab=peTab(dicttab)
        
        self.assertEqual(tab,copytab)
        copytab['energy'][15]*=0
        self.assertNotEqual(tab,copytab)
        
    def test_saving_to_db_and_loading(self):
        tab=peTab(self.p)
        tab.db()
        self.created_db_record=tab['_id']
                
        #loading using OID
        tabcopy=peTab(tab['_id'])
        self.assertEqual(tab,tabcopy)
        
        #loading using string repr of OID
        tabsecondcopy=peTab(str(tab['_id']))
        self.assertEqual(tab,tabsecondcopy)
        
        #try saving second time
        self.assertRaises(RuntimeError, tabsecondcopy.db,) 
   
        # try loading nonexisent OID
        self.assertRaises(ValueError,peTab,'000000000000000000000000') 
        
        
    def tearDown(self):
        if self.created_db_record is not None:
            mongoTable.remove(self.created_db_record) 
            
class TestExpressions(unittest.TestCase):
    def setUp(self):
        random.seed()
        self.maxlen=10000
        maxlen=self.maxlen
        length=random.randint(2,maxlen)
        self.p=peTab({
                    'randomlist':[random.random() for i in range(0,length)],
                    'randomfloat':random.random(),
                    'normcurve':np.array([
                                            random.random() 
                                            for i in range(0,length)
                                            ]),
                    'energy':np.array(range(0,length))*pq.eV,
                    'intensity':np.array([
                                                    random.random()*i
                                                    for i in range(0,length)
                                            ])*pq.dimensionless,
                    
                   'wavelength':1239.8*var('eV')*var('nm')/var('energy'),
                   'complex1':var('wavelength')/0.0000138*var('intensity')**2,
                   'complex2':var('complex1')*var('energy')/var('wavelength')**5,
                   'expset':E(var('complex1'),var('complex2'),var('wavelength')),      
               })
        self.expr=var("complex2")*var('complex1')*15
        self.exprres=(15*(((1239.8)/(0.0000138))**2/(1239.8)**5))*self.p['energy']**4*self.p['intensity']**4/(pq.eV**3*pq.nm**3)
               
    def test_eval_simple(self):
        for par,value in self.p.items():
            if not isinstance(value, (sage.symbolic.expression.Expression,Expr_set)):
                if isinstance (value,np.ndarray):
                    self.assertTrue((self.p.eval_expr(var(par))==value).all())
                else:
                    self.assertEqual(self.p.eval_expr(var(par)),value)
        
        
                
    def test_eval_elective(self):
        from pemongo.petab import EXPRESSIONS_COMPARE_PRECISION
        sum1=np.sum(self.p.eval_expr(self.expr))
        sum2=np.sum(self.exprres)
        self.assertTrue(((sum1-sum2)/(sum1+sum2))<EXPRESSIONS_COMPARE_PRECISION)       
            
suite = unittest.TestLoader().loadTestsFromTestCase(TestComparison)
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCreationSavingLoading))
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestExpressions))
#unittest.TextTestRunner(verbosity=3).run(suite)
