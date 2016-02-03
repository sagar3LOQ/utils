

#!/usr/bin/python


import os,re,gensim, string
from nltk.corpus import stopwords
from cleanData import cleanse_data



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


