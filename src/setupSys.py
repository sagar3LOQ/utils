import shutil
import model_topN as mTopNTf
import model_topn_test as mTopN
import trainWord2Vec as mWord2Vec
import convert2text as mDoc2txt
import os
import random
import math


# Global Variable declared here

raw_accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/accept"	
raw_reject_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/reject"
raw_predict_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_data/predict"

accept_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/accept"	
reject_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/reject"
predict_dir = "/home/viswanath/workspace/code_garage/conver2txt/raw_text/predict"

input_train = "/home/viswanath/workspace/code_garage/conver2txt/in_data/train"
input_test = "/home/viswanath/workspace/code_garage/conver2txt/in_data/test"
input_predict = "/home/viswanath/workspace/code_garage/conver2txt/in_data/predict"
input_total = "/home/viswanath/workspace/code_garage/conver2txt/in_data/total"

res_dir = '/home/viswanath/workspace/code_garage/conver2txt/out'
logs_dir = '/home/viswanath/workspace/code_garage/conver2txt/logs'

train_dirname = '/home/viswanath/workspace/test_resume/train'
test_dirname = '/home/viswanath/workspace/resume_data/res_dir/test'
predict_dirname = '/home/viswanath/workspace/test_resume/predict'
w2v_model_path = '/home/viswanath/workspace/code_garage/conver2txt/model/w2v_model_100v3.mod'


def chkWord2VecModel(modelPath):
	return os.path.isfile(modelPath) 

def chkTfidfModel(modelPath):
	return os.path.isfile(modelPath) 

def loadWord2Vec(modelPath):
	return mWord2Vec.load_W2Vec(modelPath)

def loadTfidfModel(path):
	return read_TfidfModel(path)

def genWord2Vec(model_name, input_data_dir, size, window, negative, sample, min_count, workers, iterations, out_model_file, is_phrased=False):
	return train(model_name, is_phrased, input_data_dir, size, window, negative, sample, min_count, workers, iterations, out_model_file)

def genTfidfModel(path):
	return get_tfidf_model(path)

def trainNtestData():
	print "====================::Resumae Shortlisting System::==============\n"
	print "Following algos are available for learning::\n(Enter respective Number to use)\n"
	print "1. TopN Model Amplified Word2vec vectors with TFIDF weights"
	print "2. TopN Model, Word2vec vectors with TFIDF to find topN words only"
	print "3. SRL based model using Word2Vec vectors and Logistic Regression"
	choice = raw_input("Enter your Choice")

	size = 100
	topN = 200


	if choice == "1":
		print "TopN Model Amplified Word2vec vectors with TFIDF weights Begins ...\n"
		gt = mTopNTf.genTopNVec(train_dirname,test_dirname,predict_dirname,w2v_model_path,size,topN)
		gt.start()

	elif choice == "2":
		print "TopN Model, Word2vec vectors with TFIDF to find topN words only Begins ...\n"
		gt = mTopN.genTopNVec(train_dirname,test_dirname,predict_dirname,w2v_model_path,size,topN)
		gt.start()

	elif choice == "3":
		print "SRL based model using Word2Vec vectors and Logistic Regression Begins ...\n"
		print "Under Integration :/"

	else:
		print "Sorry! I think its an invalid option :("

	
def NFoldTest():
	print "====================::Resumae Shortlisting System::==============\n"
	print "Following algos are available for learning::\n(Enter respective Number to use)\n"
	print "1. TopN Model Amplified Word2vec vectors with TFIDF weights"
	print "2. TopN Model, Word2vec vectors with TFIDF to find topN words only"
	print "3. SRL based model using Word2Vec vectors and Logistic Regression"
	choice = raw_input("Enter your Choice")

	size = 100
	topN = 200


	if choice == "1":
		print "TopN Model Amplified Word2vec vectors with TFIDF weights Begins ...\n"
		gt = mTopNTf.genTopNVec(train_dirname,test_dirname,predict_dirname,w2v_model_path,size,topN)
		gt.NFoldTest(iter_N=10,split =0.27)

	elif choice == "2":
		print "TopN Model, Word2vec vectors with TFIDF to find topN words only Begins ...\n"
		gt = mTopN.genTopNVec(train_dirname,test_dirname,predict_dirname,w2v_model_path,size,topN)
		gt.NFoldTest(iter_N=10,split =0.27)

	elif choice == "3":
		print "SRL based model using Word2Vec vectors and Logistic Regression Begins ...\n"
		print "Under Integration :/"

	else:
		print "Sorry! I think its an invalid option :("


def predictData():
	print "====================::Resumae Shortlisting System::==============\n"
	print "Following algos are available for learning::\n(Enter respective Number to use)\n"
	print "1. TopN Model Amplified Word2vec vectors with TFIDF weights"
	print "2. TopN Model, Word2vec vectors with TFIDF to find topN words only"
	print "3. SRL based model using Word2Vec vectors and Logistic Regression"
	choice = raw_input("Enter your Choice")

	size = 100
	topN = 200


	if choice == "1":
		print "TopN Model Amplified Word2vec vectors with TFIDF weights Begins ...\n"
		gt = mTopNTf.genTopNVec(train_dirname,test_dirname,predict_dirname,w2v_model_path,size,topN)
		gt.train_predict()

	elif choice == "2":
		print "TopN Model, Word2vec vectors with TFIDF to find topN words only Begins ...\n"
		gt = mTopN.genTopNVec(train_dirname,test_dirname,predict_dirname,w2v_model_path,size,topN)
		gt.train_predict()

	elif choice == "3":
		print "SRL based model using Word2Vec vectors and Logistic Regression Begins ...\n"
		print "Under Integration :/"

	else:
		print "Sorry! I think its an invalid option :("


def trainNtestDataRaw():
	return

def NFoldTestRaw():
	return

def predictDataRaw():
	return

def convertRawFiles2Text():
	mDoc2txt.convertDirFiles(raw_accept_dir, accept_dir)
	mDoc2txt.convertDirFiles(raw_reject_dir, reject_dir)
	mDoc2txt.convertDirFiles(raw_predict_dir, predict_dir)


def loadFiles2Input(split=0.33):

	trainExList = listFIles(input_train)
	testExList = listFIles(input_test)
	acceptExList = listFIles(accept_dir)
	rejectExList = listFIles(reject_dir)

	numTr = len(trainExList) #= listFIles(input_train)
	numTst = len(testExList) #= listFIles(input_test)
	numAcpt = len(acceptExList) #= listFIles(accept_dir)
	numRej = len(rejectExList) #= listFIles(reject_dir)

	numTstAcp = int(math.ceil(split*numAcpt))
	numTrAcp = numAcpt - numTstAcp	
#	print numTstAcp,numTrAcp

	numTstRej = int(math.ceil(split*numRej))
	numTrRej = numRej - numTstRej	
#	print numTstRej, numTrRej	

	diffTrAcp = numTrAcp - len(set(acceptExList) & set(trainExList))
	diffTstAcp = numTstAcp - len(set(acceptExList) & set(testExList))

	diffTrRej = numTrRej - len(set(rejectExList) & set(trainExList))
	diffTstRej = numTstRej - len(set(rejectExList) & set(testExList))

	acceptExList = list(set(acceptExList) - set(trainExList))
	acceptExList = list(set(acceptExList) - set(testExList))

	rejectExList = list(set(rejectExList) - set(trainExList))
	rejectExList = list(set(rejectExList) - set(testExList))


	if diffTrAcp>0:		
		TrAcpFile = set(random.sample(acceptExList,diffTrAcp))
		acceptExList = list(set(acceptExList) - set(TrAcpFile))
		for f in TrAcpFile:
	#		print f
	#		print "\n\n"
#			moveData(accept_dir +"/"+f,input_train)
#			moveData(accept_dir +"/"+f,input_total)

			moveData(os.path.join(accept_dir,f),input_train)
			moveData(os.path.join(accept_dir,f),input_total)


	if diffTstAcp>0:		
		TstAcpFile = set(random.sample(acceptExList,diffTstAcp))
		for f in TstAcpFile:
	#		print f
	#		print "\n\n"
#			moveData(accept_dir +"/"+f,input_test)
#			moveData(accept_dir +"/"+f,input_total)

			moveData(os.path.join(accept_dir,f),input_test)
			moveData(os.path.join(accept_dir,f),input_total)


	if diffTrRej>0:		
		TrRejFile = set(random.sample(rejectExList,diffTrRej))
		rejectExList = list(set(rejectExList) - set(TrRejFile))
		for f in TrRejFile:
	#		print f
	#		print "\n\n"
#			moveData(reject_dir +"/"+f,input_train)
#			moveData(reject_dir +"/"+f,input_total)

			moveData(os.path.join(reject_dir,f),input_train)
			moveData(os.path.join(reject_dir,f),input_total)


	if diffTstRej>0:		
		TstRejFile = set(random.sample(rejectExList,diffTstRej))
		for f in TstRejFile:
	#		print f
	#		print "\n\n"
#			moveData(reject_dir +"/"+f,input_test)
#			moveData(reject_dir +"/"+f,input_total)

			moveData(os.path.join(reject_dir,f),input_test)
			moveData(os.path.join(reject_dir,f),input_total)


	predictList = listFIles(predict_dir)
	predictExList = listFIles(input_predict)

	predictList = set(predictList) - set(predictExList)

	for f in predictList:
	#	print f
#		moveData(predict_dir +"/"+f,input_predict)
#		moveData(predict_dir +"/"+f,input_total)

		moveData(os.path.join(predict_dir,f),input_predict)
		moveData(os.path.join(predict_dir,f),input_total)




def listFIles(dirname):
	files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
	return files

def moveData(src,dest):
	shutil.copy(src,dest)

def clearDir(Dir):
	for root, dirs, files in os.walk(Dir):
		for name in files:
			os.remove(os.path.join(root, name))

def cleanShuffle():

	
	return

if __name__ == '__main__':
	convertRawFiles2Text()
	loadFiles2Input()
