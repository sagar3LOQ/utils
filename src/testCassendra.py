from cqlengine import columns
from cqlengine.models import Model
from cqlengine import connection
from cqlengine.management import sync_table
from convert2text_IterText import convertFiles2TextIterWrap
from cqlengine.exceptions import LWTException
from timeMeasure import timeMeasure

# Define a model for existing table in Cassandra Database
class RUser(Model):
  fname = columns.Text()
  md5 = columns.Text(primary_key=True)
  label = columns.Text()
  text = columns.Text()
 


# Class for accessing Cassandra Database

class CassandraClient:

    def __init__(self):
        self.err= None


    def connectCassandra(self):
        connection.setup(['127.0.0.1'], "resumeks")
        sync_table(RUser)


    def insertData(self, dataGen):
        for data in dataGen:

            meta = data[1].split("_")

            try:

                RUser.if_not_exists().create(fname = data[0],md5 =meta[0], label =meta[1], text = data[2])
            except  LWTException as e:

                print e # existing object


    def deleteData(self):
        return

    def updateData(self):
        return

    def getAllData(self):
        return RUser.objects()

    def printData(self,cur):
        for d in cur:
            print d
            print "\n\n"





if __name__ == '__main__':


    CasCl = CassandraClient()


    tm = timeMeasure()


# Connection to Cassendra DB
    try:

  
        tm.start()

        # Connect to the resumeKS keyspace on our cluster running at 127.0.0.1
        connection.setup(['127.0.0.1'], "resumeKS")

        # Sync your model with your cql table
   #     sync_table(RUser)
        t1 = tm.stop()
        print "Connected successfully!!!"
    except :
        print "Could not connect to Cassandra" 



    print "Started code"
#    accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/accept"
    accept_dir = "/home/viswanath/Downloads/total_resume"
    accept_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/accept"




    dataGen = convertFiles2TextIterWrap(accept_dir)


    tm.start()
    CasCl.insertData(dataGen)

    t2 = tm.stop()


    tm.start()
    
    CasCl.getAllData()
    t3 = tm.stop()


    print "time to connect:: " + str(t1) + "sec"
    print "time to insert:: " + str(t2) + "sec"
    print "time to fetchAll:: " + str(t3) + "sec"
