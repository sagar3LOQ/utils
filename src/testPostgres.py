import psycopg2
from convert2text_IterText import convertFiles2TextIterWrap
from timeMeasure import timeMeasure



# Class for accessing PostgreSQL Database
class PostgreClient:

    def __init__(self):
        self.conn = None
        self.cur = None

    def connectPostgre(self):
        self.conn = psycopg2.connect(database="testdb", user="postgres", password="sin2win", host="127.0.0.1", port="5432")
        return self.conn

    def getDBPointer(self):
        self.cur = self.conn.cursor()
        return  self.cur


    def insertData(self,dataGen):
        for data in dataGen:

            meta = data[1].split("_")

            query = "INSERT INTO RESUME (md5, fname, label, text1) \
                     SELECT '"+ meta[0] + "', '" + data[0] +"', '" + meta[1] + "', '" + data[2] + "' \
                     WHERE NOT EXISTS (SELECT 1 FROM RESUME WHERE md5= '" + meta[0] + "')"
        
            self.cur.execute(query);


    def deleteData(self):
        return

    def updateData(self):
        return


    def getAllData(self):
        self.cur.execute("SELECT md5, fname, label, text1  from RESUME")
        return self.cur.fetchall()

    def printData(self,rows):
        for row in rows:
            print "MD5 = ", row[0]
            print "FNAME = ", row[1]
            print "LABEL = ", row[2]
            print "TEXT = ", row[3], "\n\n"



if __name__ == '__main__':

# Connection to Postgresql DB

    try:

        tm = timeMeasure()

        tm.start()
        PgCL = PostgreClient()
        PgCL.connectPostgre()
        t1 = tm.stop()
        print "Opened database successfully"     #   print "Connected successfully!!!"

    except :

        print "Could not connect to Postgre" 


    PgCL.getDBPointer()

    print "Started code"
#    accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/accept"
    accept_dir = "/home/viswanath/Downloads/total_resume"
    accept_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/accept"


    dataGen = convertFiles2TextIterWrap(accept_dir)


    tm.start()
    PgCL.insertData(dataGen)


    PgCL.conn.commit()
    t2 = tm.stop()

    tm.start()
    PgCL.getAllData()
    t3 = tm.stop()

#    PgCL.printData(PgCL.getAllData())


    PgCL.conn.close()
    print "time to connect:: " + str(t1) + "sec"
    print "time to insert:: " + str(t2) + "sec"
    print "time to fetchAll:: " + str(t3) + "sec"


