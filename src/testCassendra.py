from cqlengine import columns
from cqlengine.models import Model
from cqlengine import connection
from cqlengine.management import sync_table
from convert2text_IterText import convertFiles2TextIterWrap
from cqlengine.exceptions import LWTException


# Define a model
class RUser(Model):
  fname = columns.Text()
  md5 = columns.Text(primary_key=True)
  label = columns.Text()
  text = columns.Text()
 
  def __repr__(self):
    return '%s %s' % (self.fname, self.md5)


if __name__ == '__main__':
# Connection to Cassendra DB
    try:

        # Connect to the resumeKS keyspace on our cluster running at 127.0.0.1
        connection.setup(['127.0.0.1'], "resumeKS")

        # Sync your model with your cql table
        sync_table(RUser)
        print "Connected successfully!!!"
    except :
        print "Could not connect to Cassandra" 



    print "Started code"
    accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/accept"
    accept_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/accept"
#    convertDirFiles(accept_dir,accept_out)
    dataGen = convertFiles2TextIterWrap(accept_dir)
    for data in dataGen:

        meta = data[1].split("_")

        try:

            RUser.if_not_exists().create(fname = data[0],md5 =meta[0], label =meta[1], text = data[2])
        except  LWTException as e:

            print e # existing object

    cur = RUser.objects()
    
    for d in cur:
        print d
        print "\n\n"
