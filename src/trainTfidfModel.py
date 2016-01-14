#!/usr/bin/python


import os,re,gensim, string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import pickle


def cleanse_data(text):

##
	##  Remove all non relevent symbols and get the text
	## that can be used to clean our data with noise
##

#	print "cleansing"
	temp = re.sub(r'[^\x00-\x7F]+',' ', text)
	temp = re.sub(r'(\d+(\s)?(yrs|year|years|Yrs|Years|Year|yr))'," TIME ",temp)
	temp = re.sub(r'[\w\.-]+@[\w\.-]+'," EMAIL ",temp)
	temp = re.sub(r'(((\+91|0)?( |-)?)?\d{10})',' MOBILE ',temp)
	temp = re.sub(r"[\r\n]+[\s\t]+",'\n',temp)	
	wF = set(string.punctuation) - set(["+"])
	for c in wF:
        	temp =temp.replace(c," ")	

	return temp.lower()


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

class Sentences(object):
	def __init__(self, dirname):
		self.dirname = dirname
		self.flist = []
	
	def __iter__(self):

		for fname in os.listdir(self.dirname):
			self.flist.append(fname)

			f = open(os.path.join(self.dirname, fname))
			text = str.decode(f.read(), "UTF-8", "ignore")
			text = cleanse_data(text)
			yield text


def get_tfidf_model(dirname):

	data = Sentences(dirname)
	tfidf_vectorizer = TfidfVectorizer(stop_words='english')
	tfidf_matrix = tfidf_vectorizer.fit_transform(data)
	mat_array = tfidf_matrix.toarray()
	fn = tfidf_vectorizer.get_feature_names()

	return tfidf_vectorizer


def save_TfidfModel(tfidf_model,modelPath):
	timestr = time.strftime("%Y_%m_%d_%H%M%S")
	tfidf_mod = modelPath +"/"+timestr + "_tfidf.mod"
	with open(tfidf_mod, 'wb') as fin:
		pickle.dump(tfidf_model, fin)
	return tfidf_mod

def read_TfidfModel(tf_path):
	return pickle.load(open( tf_path, "rb" ))

def main():
	fn = '/home/viswanath/workspace/code_garage/conver2txt/in_data/total'
	mp = '/home/viswanath/workspace/code_garage/conver2txt/model'
	tfmod = get_tfidf_model(fn)
	mat1 = tfmod.idf_
	tfp = save_TfidfModel(tfmod,mp)

	tmod = read_TfidfModel(tfp)
	mat2 = tmod.idf_
	if (mat1==mat2).all(): 
		print "mission successful :)"
	else:
		print "your code sucks :/"

if __name__ == "__main__": main()
