# This file was *autogenerated* from the file ./wks/sage-global/pemongo/ssh_server.sage.
from sage.all_cmdline import *   # import sage library
_sage_const_2 = Integer(2); _sage_const_1 = Integer(1); _sage_const_0 = Integer(0); _sage_const_5 = Integer(5); _sage_const_5022 = Integer(5022); _sage_const_1024 = Integer(1024); _sage_const_300 = Integer(300); _sage_const_50 = Integer(50)#!/usr/bin/env sage

from twisted.cred import portal, checkers
from twisted.conch import error, avatar
from twisted.conch.checkers import SSHPublicKeyDatabase
from twisted.conch.ssh import factory, userauth, connection, keys, session
from twisted.internet import reactor, protocol, defer
from twisted.python import log
from zope.interface import implements
import sys
from getopt import getopt,GetoptError
from tempfile import mkdtemp
from base64 import b64encode,b64decode
from os import listdir
from os.path import isfile, join
from subprocess import call,STDOUT
from threading import Thread

log.startLogging(sys.stderr)

MAX_BYTES_PER_SESSION=_sage_const_1024 *_sage_const_1024 *_sage_const_5  #5 megabytes
MAX_FILES_PER_SESSION=_sage_const_50 
MAX_FILESIZE_BYTES=_sage_const_1024 *_sage_const_1024 *_sage_const_5  #5 megabytes
MAX_EXEC_TIME_SECONDS=_sage_const_300  # 5 min

def setopts(argv):

    global MAX_BYTES_PER_SESSION
    global MAX_FILES_PER_SESSION
    global MAX_FILESIZE_BYTES
    global MAX_EXEC_TIME_SECONDS
    hlpmsg='''options:
     -b maxbytes\t max bytes of encoded text sent/received per session (default '''+str(MAX_BYTES_PER_SESSION)+''')
     -f maxfiles\t max files send per session (default '''+str(MAX_FILES_PER_SESSION)+''')
     -s maxsize \t max file size to send  (default '''+str(MAX_FILESIZE_BYTES)+''')
     -t maxtime \t max computation time  (default '''+str(MAX_EXEC_TIME_SECONDS)+''')'''
    try:
      opts, args = getopt(argv,"hb:f:s:t:",["maxbytes=","maxfiles=","maxsize=","maxtime="])
    except GetoptError:
      print hlpmsg
      sys.exit(_sage_const_2 )
    for opt, arg in opts:
      if opt == '-h':
         print hlpmsg
         sys.exit()
      elif opt in ("-b", "--maxbytes"):
         MAX_BYTES_PER_SESSION = int(arg)
      elif opt in ("-f", "--maxfiles"):
         MAX_FILES_PER_SESSION = int(arg)
      elif opt in ("-s", "--maxsize"):
         MAX_FILESIZE_BYTES = int(arg)
      elif opt in ("-t", "--maxtime"):
         MAX_EXEC_TIME_SECONDS = int(arg)


"""
Example of running another protocol over an SSH channel.
log in with username "user" and password "password".
"""

class ExampleAvatar(avatar.ConchUser):

    def __init__(self, username):
        avatar.ConchUser.__init__(self)
        self.username = username
        self.channelLookup.update({'session':session.SSHSession})

class ExampleRealm:
    implements(portal.IRealm)

    def requestAvatar(self, avatarId, mind, *interfaces):
        return interfaces[_sage_const_0 ], ExampleAvatar(avatarId), lambda: None

class EchoProtocol(protocol.Protocol):
    """this is our example protocol that we will run over SSH
    """
    def connectionMade(self):
        print 'connection established - initializing'
        self.tmpdir=mkdtemp()
        self.iodir=join(self.tmpdir,".io")
        os.makedirs(self.iodir) 
        self.inputfile=join(self.iodir,"input.b64")
        self.execfile=join(self.iodir,"input.sage")
        self.textfile=join(self.iodir,"output.txt")
        self.outputfile=join(self.iodir,"output.b64")
        self.exec_process=None
        
        print self.tmpdir
        print self.iodir
        print self.inputfile
        print self.execfile
        print self.textfile
        print self.outputfile
        
    def dataReceived(self, data):
        print 'data received'
        # terminate on ^C
        if data.find('\x03')!=-_sage_const_1 :
            self.cleanup() 
            self.transport.loseConnection() 
        if self.exec_process is not None:  
            #if self.exec_process.is_alive():
            return # already started - no more data accepted
                  
        # stop data receiving on EOF    
        eofpos=data.find('\x04') 
        if eofpos==-_sage_const_1 :
            print 'writing received data to file'
            with open(self.inputfile,'a') as f:
                f.write(data)
            self.check_long_data()
            return
        # got EOF in data - start processing
        print 'finishing writing received data to file'
        with open(self.inputfile,'a') as f:
            f.write(data[_sage_const_0 :eofpos])
        if self.check_long_data():
            return
        print 'starting decodind'
        # leave the decoding to base64 program

        with open(self.execfile,'w') as out:
            retval=call(["base64", "-d", self.inputfile],stdout=out,stderr=STDOUT)  
        if retval!=_sage_const_0 :
            self.transport.write(b64encode('raise RuntimeException("Input decoding failed")\n'))              
            self.cleanup()
            self.transport.loseConnection()
            return
        print 'decoding complete, executing subprocess'
        # create subprocess object. Note that exec_process.computing is set here    
        self.exec_process=ExecutorProcess(self)  
        # set up watchdog - a new thread in tne main process
        wdt=Thread(target=self.watchdog)
        wdt.daemon = True
        # then run the subprocess                 
        self.exec_process.start()
        # and the watchdog
        wdt.start()
    
    def check_long_data(self):
        if  os.path.getsize(self.inputfile) > MAX_BYTES_PER_SESSION:
            print 'too long data file received'
            self.transport.write(b64encode('raise RuntimeException("Too long data received")\n'))
            self.cleanup()
            self.transport.loseConnection() 
            return True
        return False
        
    def on_process_finished(self):
        
        print 'subprocess exited, encoding stdout'
        mypath=self.exec_process.tmpdir
        onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
        
        # encode stdout
        with open(self.outputfile,'w') as out:
            retval=call(["base64", "-w 0", self.textfile],stdout=out,stderr=STDOUT)  
        if retval!=_sage_const_0 :
            self.transport.write(b64encode('raise RuntimeException("Output encoding failed")\n'))              
            self.cleanup()
            self.transport.loseConnection()
            return
        print 'stdout encoded, sending files'
        #send stdout    
        if os.path.getsize(self.outputfile) < MAX_BYTES_PER_SESSION:
            with open(self.outputfile,'r') as f:
                self.transport.write(f.read()+'\n')
        else:
            self.transport.write(b64encode('raise RuntimeException("Too long output")\n'))
            self.cleanup(False) # we cannot kill process from itself
            self.transport.loseConnection()
            return
            
        #read files, encode them and push into ssh
        for i,filename in enumerate(onlyfiles):
            if i > MAX_FILES_PER_SESSION:
                break 
            if os.path.getsize(os.path.join(mypath,filename))>MAX_FILESIZE_BYTES:
                self.transport.write(b64encode('raise RuntimeException("Too big file ' + filename + '")\n'))
            else:
            # the following is bad for large files. We however set a limit.
                with open(os.path.join(mypath,filename),'r') as f:
                    self.transport.write(b64encode(filename)+'\n'+b64encode(f.read())+'\n')

        self.cleanup(False) # we cannot kill process from itself
        self.transport.loseConnection()
        
    def cleanup(self,killproc=True):
        print 'cleaning up'
        if killproc:
            if self.exec_process is not None:
            # terminate if still alive
                if self.exec_process.is_alive():
                        self.exec_process.terminate()
            
        from shutil import rmtree
        rmtree(self.tmpdir)
            
    def watchdog(self):
        from time import clock
        start = clock()
        print 'watchdog on'
        while(True):
            #try:
            self.exec_process.join(_sage_const_1 ) 
            #except AssertionError:
                #sleep(0.25)      
            if self.exec_process is None:
                print 'the subprocess ref is None - watchdog off'
                # this means we launched watchdog too early
                raise RuntimeError('None is not a process to watchdog')
                return
            if self.exec_process.is_alive()==False:# and self.exec_process.was_alive :
                print 'the subprocess finished - watchdog off'
                self.on_process_finished()
                return
            #if self.exec_process.computing is False:
            #    print 'computation finished - watchdog off'
            #    return
            if (clock() - start)>MAX_EXEC_TIME_SECONDS:
                self.exec_process.terminate() # now the on_process_finished will not be called
                print 'watchdog triggered - computation terminated'
                self.transport.write(b64encode('raise RuntimeException("Computation takes too long time")'))
                self.cleanup()
                self.transport.loseConnection()
                return
                        

publicKey = 'ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAGEArzJx8OYOnJmzf4tfBEvLi8DVPrJ3/c9k2I/Az64fxjHf9imyRJbixtQhlH9lfNjUIx+4LmrJH5QNRsFporcHDKOTwTTYLh5KmRpslkYHRivcJSkbh/C+BR3utDS555mV'

privateKey = """-----BEGIN RSA PRIVATE KEY-----
MIIByAIBAAJhAK8ycfDmDpyZs3+LXwRLy4vA1T6yd/3PZNiPwM+uH8Yx3/YpskSW
4sbUIZR/ZXzY1CMfuC5qyR+UDUbBaaK3Bwyjk8E02C4eSpkabJZGB0Yr3CUpG4fw
vgUd7rQ0ueeZlQIBIwJgbh+1VZfr7WftK5lu7MHtqE1S1vPWZQYE3+VUn8yJADyb
Z4fsZaCrzW9lkIqXkE3GIY+ojdhZhkO1gbG0118sIgphwSWKRxK0mvh6ERxKqIt1
xJEJO74EykXZV4oNJ8sjAjEA3J9r2ZghVhGN6V8DnQrTk24Td0E8hU8AcP0FVP+8
PQm/g/aXf2QQkQT+omdHVEJrAjEAy0pL0EBH6EVS98evDCBtQw22OZT52qXlAwZ2
gyTriKFVoqjeEjt3SZKKqXHSApP/AjBLpF99zcJJZRq2abgYlf9lv1chkrWqDHUu
DZttmYJeEfiFBBavVYIF1dOlZT0G8jMCMBc7sOSZodFnAiryP+Qg9otSBjJ3bQML
pSTqy7c3a2AScC/YyOwkDaICHnnD3XyjMwIxALRzl0tQEKMXs6hH8ToUdlLROCrP
EhQ0wahUTCk1gKA4uPD6TMTChavbh4K63OvbKg==
-----END RSA PRIVATE KEY-----"""


class InMemoryPublicKeyChecker(SSHPublicKeyDatabase):

    def checkKey(self, credentials):
        return credentials.username == 'user' and \
            keys.Key.fromString(data=publicKey).blob() == credentials.blob

class ExampleSession:
    
    def __init__(self, avatar):
        """
        We don't use it, but the adapter is passed the avatar as its first
        argument.
        """

    def getPty(self, term, windowSize, attrs):
        pass
    
    def execCommand(self, proto, cmd):
        pass # do not execute anything
        
    def openShell(self, trans):
        ep = EchoProtocol()
        ep.makeConnection(trans)
        trans.makeConnection(session.wrapProtocol(ep))

    def eofReceived(self):
        pass

    def closed(self):
        pass

from twisted.python import components
components.registerAdapter(ExampleSession, ExampleAvatar, session.ISession)

class ExampleFactory(factory.SSHFactory):
    publicKeys = {
        'ssh-rsa': keys.Key.fromString(data=publicKey)
    }
    privateKeys = {
        'ssh-rsa': keys.Key.fromString(data=privateKey)
    }
    services = {
        'ssh-userauth': userauth.SSHUserAuthServer,
        'ssh-connection': connection.SSHConnection
    }
#from threading import Thread
from multiprocessing import Process

class ExecutorProcess(Process):
    def __init__(self, parent):
        self.execfile=parent.execfile
        self.textfile=parent.textfile
        self.tmpdir=parent.tmpdir
        self.parent=parent
        self.computing=True
        #self.was_alive=False
        Process.__init__(self)
    def run(self):
        #self.was_alive=True
        from StringIO import StringIO
        from traceback import format_exc
        import sys 
        from os import chdir,getcwd       
        print 'subprocess started'
                
        old_stdout = sys.stdout    # Store the reference                    
        olddir=getcwd() 
        chdir(self.tmpdir)
        with open(self.textfile,'w') as out:
            sys.stdout=out
            try: 
                load(self.execfile)
                #exec compile(preparse(self.data),'', 'exec') in globals()
            except Exception as e:
                print 'Exception:' + e.message + '\r\n'
                print format_exc() + '\r\n'            
            finally: 
                self.computing=False          
                sys.stdout = old_stdout   # Redirect again the std output to screen
                chdir(olddir)
            #sys.stderr = old_stderr    
    
        #self.parent.on_process_finished(self)
        

        
portal = portal.Portal(ExampleRealm())
passwdDB = checkers.InMemoryUsernamePasswordDatabaseDontUse()
passwdDB.addUser('user', '')
portal.registerChecker(passwdDB)
portal.registerChecker(InMemoryPublicKeyChecker())
ExampleFactory.portal = portal

def startserver(port=_sage_const_5022 ):
    reactor.listenTCP(port, ExampleFactory())
    reactor.run()

if __name__ == '__main__':
    setopts(sys.argv[_sage_const_1 :])
    startserver(port=_sage_const_5022 )

