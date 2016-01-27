

#!/usr/bin/python


import os,re,gensim, string
from nltk.corpus import stopwords


def cleanse_data(text):

##
	##  Remove all non relevent symbols and get the text
	## that can be used to clean our data with noise
##

	text = re.sub(r'[^\x00-\x7F]+',' ', text)
	text = re.sub(r'(\d+[\+\s]*(Years|years|Year|year|Yrs|yrs|yr))',"year",text)
	text = re.sub(r'[\w\.-]+@[\w\.-]+'," EMAIL ",text)
	text = re.sub(r'(((\+91|0)?( |-)?)?\d{10})',' MOBILE ',text)
	text = re.sub(r"[\r\n]+[\s\t]+",'\n',text)	

	text = re.sub(r"\.[\s\t\n]+",'\n',text)	

	wF = set(string.punctuation) - set(["+"])

	for c in wF:
        	text =text.replace(c," ")	

	return text.lower()



def scan_file(dir_name):

##
	##  scan every file in a directory 
	##  and extract text from it
##
	for fname in os.listdir(dir_name):
		fp = open(os.path.join(dir_name, fname),"r")
		text = fp.read()
		yield cleanse_data(text)

def save_file(text,fname):

##
	##  save the files 
	##  text is text and fname is the path and name of the file to be save 
##
	fp = open(fname,"a")
#	fp.write("\n\n ********************** NEW FILE **********************\n\n")
	fp.write(text)
	fp.write("\n")
	fp.close()

def words(stringIterable):

    #upcast the argument to an iterator, if it's an iterator already, it stays the same
    lineStream = iter(stringIterable)
    for line in lineStream: #enumerate the lines
        for word in line.split(" \n\t"): #further break them down
            yield word

class SentencesTokenise(object):
	def __init__(self, dirname):
		self.dirname = dirname
	
	def __iter__(self):

		for fname in os.listdir(self.dirname):

			for line in open(os.path.join(self.dirname, fname)):
				
				line = cleanse_data(line)
				yield line.lower().split()


class Sentences(object):
	def __init__(self, dirname):
		self.dirname = dirname
		self.flist = []
	
	def __iter__(self):
#		count = 0
		#flist = []
		for fname in os.listdir(self.dirname):
			self.flist.append(fname)
#			count = count +1
#			print fname, count
			f = open(os.path.join(self.dirname, fname))
			text = str.decode(f.read(), "UTF-8", "ignore")
			text = cleanse_data(text)
			yield text.lower()

