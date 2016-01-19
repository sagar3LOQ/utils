import pymongo
from convert2text_IterText import convertFiles2TextIterWrap
from timeMeasure import timeMeasure



class MongoClient:

    def __init__(self):
        self.conn = None
        self.collection = None
        self.db = None
   

    def connectMongo(self):
        try:
            self.conn = pymongo.MongoClient()
            return self.conn
          #  print "Connected successfully!!!"
        except pymongo.errors.ConnectionFailure, e:
            print "Could not connect to MongoDB: %s" % e 

    def getDB(self,conn):
        if conn==None: return
        self.db =  conn.resumeDB
        return  self.db 

    def getCollection(self,db):
        if db==None: return
        self.collection =  db.resume_collection
        return  self.collection 

    def autoConnect(self):
        self.connectMongo()
        self.getDB(self.conn)
        self.getCollection(self.db)
        if self.collection == None: return 0
        return 1

    def insertData(self, dataGen):
        if dataGen==None: return
        for data in dataGen:        
            rowData = {}
            rowData["fname"] = data[0]
            meta = data[1].split("_")
            rowData["md5"] = meta[0]
            rowData["label"] = meta[1]
            rowData["text"] = data[2]
            if self.collection.find({'md5': meta[0]}).limit(1).count() > 0: continue
            self.collection.insert(rowData)


    def deleteData(self):
        return

    def updateData(self):
        return

    def getAllData(self):
        return self.collection.find()

    def printData(self,cur):
        for d in cur:
            print d
            print "\n\n"


if __name__ == '__main__':

    
# Connection to Mongo DB

    tm = timeMeasure()

    tm.start()

    pyMonObj = MongoClient()
    err = pyMonObj.autoConnect()

    t1 = tm.stop()
    if err == 0:
        print "Connection Failed...\n"

    print "Processing started...\n"

 #   accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/accept"
    accept_dir = "/home/viswanath/workspace/code_garage/te"
    accept_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/accept"

#    convertDirFiles(accept_dir,accept_out)

    dataGen = convertFiles2TextIterWrap(accept_dir)

    tm.start()

    pyMonObj.insertData(dataGen)


    t2 = tm.stop()


    print pyMonObj.conn.database_names()

    print pyMonObj.db.collection_names()


    tm.start()
    pyMonObj.getAllData()
    t3 = tm.stop()

    print "time to connect:: " + str(t1) + "sec"
    print "time to insert:: " + str(t2) + "sec"
    print "time to fetchAll:: " + str(t3) + "sec"





