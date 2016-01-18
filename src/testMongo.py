import pymongo
from convert2text_IterText import convertFiles2TextIterWrap

if __name__ == '__main__':
# Connection to Mongo DB
    try:
        conn=pymongo.MongoClient()
        print "Connected successfully!!!"
    except pymongo.errors.ConnectionFailure, e:
        print "Could not connect to MongoDB: %s" % e 

    db = conn.resumeDB
    print db

    collection = db.resume_collection

    print "Started code"
    accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/accept"
    accept_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/accept"
#    convertDirFiles(accept_dir,accept_out)
    dataGen = convertFiles2TextIterWrap(accept_dir)
    for data in dataGen:
        rowData = {}
        rowData["fname"] = data[0]
        meta = data[1].split("_")
        rowData["md5"] = meta[0]
        rowData["label"] = meta[1]
        rowData["text"] = data[2]
        if collection.find({'md5': meta[0]}).limit(1).count() > 0: continue
        collection.insert(rowData)
#        print "\n\n"

    print conn.database_names()

    print db.collection_names()

    cur = collection.find()
    
    for d in cur:
        print d
        print "\n\n"


