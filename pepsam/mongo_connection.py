#This file detects if we are working with or without database

try:
    from .mongoDBdetails import mongoConnection, mongoDB, petabTable, scriptTable
except ImportError:
    mongoConnection=mongoDB=petabTable=scriptTable=None
