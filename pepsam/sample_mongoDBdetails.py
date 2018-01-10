
from pymongo import MongoClient

mongoConnection = MongoClient('127.0.0.1', 27017,tz_aware=True)
#connect to a database named 'test'
mongoDB = mongoConnection.test
#open a table with peTab instances
petabTable=mongoDB.petabs
#open a table with peScript instances
scriptTable=mongoDB.pescripts
