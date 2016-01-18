import psycopg2
from convert2text_IterText import convertFiles2TextIterWrap




if __name__ == '__main__':
# Connection to Postgresql DB
    try:

        # Connect to the testdb database on our postgres  running at 127.0.0.1:5432
        conn = psycopg2.connect(database="testdb", user="postgres", password="sin2win", host="127.0.0.1", port="5432")

        print "Opened database successfully"     #   print "Connected successfully!!!"
    except :
        print "Could not connect to Postgre" 


    cur = conn.cursor()
    print "Started code"
    accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/accept"
    accept_out = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/accept"
#    convertDirFiles(accept_dir,accept_out)
    dataGen = convertFiles2TextIterWrap(accept_dir)
    for data in dataGen:

        meta = data[1].split("_")

        query = "INSERT INTO RESUME (md5, fname, label, text1) \
                     SELECT '"+ meta[0] + "', '" + data[0] +"', '" + meta[1] + "', '" + data[2] + "' \
                     WHERE NOT EXISTS (SELECT 1 FROM RESUME WHERE md5= '" + meta[0] + "')"
        
        cur.execute(query);
        
        conn.commit()
        print "Records created successfully";


        cur.execute("SELECT md5, fname, label, text1  from RESUME")
        rows = cur.fetchall()
        for row in rows:
            print "MD5 = ", row[0]
            print "FNAME = ", row[1]
            print "LABEL = ", row[2]
            print "TEXT = ", row[3], "\n\n"

        print "Operation done successfully";

    conn.close()
    

