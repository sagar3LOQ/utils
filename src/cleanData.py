
import re
import string


def cleanse_data(text):

##
	##  Remove all non relevent symbols and get the text
	## that can be used to clean our data with noise
##

	text = re.sub(r'[^\x00-\x7F]+',' ', text)
	text = re.sub(r'(\d+(\s)?(yrs|year|years|Yrs|Years|Year|yr))'," TIME ",text)
	text = re.sub(r'[\w\.-]+@[\w\.-]+'," EMAIL ",text)
	text = re.sub(r'(((\+91|0)?( |-)?)?\d{10})',' MOBILE ',text)
	text = re.sub(r"[\r\n]+[\s\t]+",'\n',text)	

	text = re.sub(r"\.[\s\t\n]+",'\n',text)	

	wF = set(string.punctuation) - set(["+"])

	for c in wF:
        	text =text.replace(c," ")	

	return text.lower()

