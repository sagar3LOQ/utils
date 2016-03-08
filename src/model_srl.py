import sys
import gensim
from gensim.models import Word2Vec
from sklearn.externals import joblib
from gensim import utils, matutils
import scipy
import numpy as np
import fastcluster
import scipy.cluster.hierarchy
import scipy.cluster.hierarchy as sch 
import string
from pprint import pprint
from configobj import ConfigObj
import traceback
import re, os, ast
import logging
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score
from sklearn import datasets, linear_model, cross_validation
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import TfidfVectorizer
from model_srl_utils import DocumentFeatures
from sentenceUtil import Sentences
import numpy as Math
import pylab as Plot
from senna_py import srl_extract
from cleanData import cleanse_data



res_dir = '/home/viswanath/workspace/code_garage/conver2txt/out'
logs_dir = '/home/viswanath/workspace/code_garage/conver2txt/logs'


### class for training data model 

class TrainData():

	def __init__(self):
		pass

    # load word2vec model
	def load_w2vmodel(self,model):
		return gensim.models.Word2Vec.load(model)

    # generate TF-IDF model from dirname files
	def get_tfidf_model(self, dirname):
		
		data = Sentences(dirname)
		tfidf_vectorizer = TfidfVectorizer()
		tfidf_matrix_train = tfidf_vectorizer.fit_transform(data)
		return tfidf_vectorizer
	
	# generate document vectors 
	def train_model(self, dirname, w2v_model_path,ndim):
		
		tfidf_model = self.get_tfidf_model(dirname)
		w2v_model = self.load_w2vmodel(w2v_model_path)
		trd = DocumentFeatures()
		wt_vect_data = []
		label_data = []
		fn = []
		for fname in os.listdir(dirname):
			f = open(os.path.join(dirname, fname))
			text = str.decode(f.read(), "UTF-8", "ignore")
			text = cleanse_data(text)	
			print "processsing ::" + fname
			VA0, VA1 =srl_extract(text)
			sent_vect = trd.get_sent_circconv_vec(text, w2v_model, ndim, 'tfidf', tfidf_model,VA0, VA1)
			if fname[-10:-4] == 'accept':
				label = 1
			else:
				label = 0
			fn.append(fname)
			wt_vect_data.append(sent_vect[0])
			label_data.append(label)

		return wt_vect_data, label_data, fn


### class for extracting data on predict data 

class PredictData():

	def __init__(self):
		pass

	def load_w2vmodel(self,model):
		return gensim.models.Word2Vec.load(model)

	def get_tfidf_model(self, dirname):
		
		data = Sentences(dirname)
		tfidf_vectorizer = TfidfVectorizer()
		tfidf_matrix_train = tfidf_vectorizer.fit_transform(data)
		return tfidf_vectorizer
	
	def train_model(self, dirname, w2v_model_path,ndim):
		
		tfidf_model = self.get_tfidf_model(dirname)
		w2v_model = self.load_w2vmodel(w2v_model_path)
		trd = DocumentFeatures()
		wt_vect_data = []
		label_data = []
		fn = []
		for fname in os.listdir(dirname):
			f = open(os.path.join(dirname, fname))
			text = str.decode(f.read(), "UTF-8", "ignore")
			text = cleanse_data(text)	
			print "processsing ::" + fname
			VA0, VA1 =srl_extract(text)
			sent_vect = trd.get_sent_circconv_vec(text, w2v_model, ndim, 'tfidf', tfidf_model,VA0, VA1)

			fn.append(fname)
			wt_vect_data.append(sent_vect[0])

		return wt_vect_data, fn


#### generate SRL structure and calculate vector based on Circulation convolution of SRL structre based on paper mentioned  		
		
class genSRLVec():

	def __init__(self,train_dirname,test_dirname,predict_dirname,w2v_model_path,size):
		self.train_dirname = train_dirname
		self.test_dirname = test_dirname
		self.w2v_model_path = w2v_model_path
		self.size = size
		self.x_wt = []
		self.Ylabels = []
		self.xTest_wt = []
		self.Y_test = []
		self.Y_pred = []
		self.result = []
		self.fn_train = []
		self.fn_test = []
		self.predict_dirname = predict_dirname
		self.xPred_wt = []		

	def start(self):

		td = TrainData()
		self.x_wt, self.Ylabels,self.fn_train = td.train_model(self.train_dirname, self.w2v_model_path,self.size)
		self.xTest_wt, self.Y_test,self.fn_test = td.train_model(self.test_dirname, self.w2v_model_path,self.size)	

		
		print "###################### LR Training ###########################"
		logit = LogisticRegression(C=1.0).fit(self.x_wt, self.Ylabels)

		print "####################### LR Prediction ##########################"
		self.Y_pred = logit.predict(self.xTest_wt)

		self.result = self.getAnalysis(self.Y_test,self.Y_pred)



	def train_predict(self):

		td = TrainData()
		self.x_wt, self.Ylabels,self.fn_train  = td.train_model( self.train_dirname, self.w2v_model_path,self.size)

		# For extracting data for predict set
		pd = PredictData()
		self.xPred_wt,self.fn_test = pd.predict_model(self.predict_dirname, self.w2v_model_path, self.size)


		print "###################### LR Training ###########################"
		logit = LogisticRegression(C=1.0).fit(self.x_wt, self.Ylabels)

		print "####################### LR Prediction ##########################"
		self.Y_pred = logit.predict(self.xPred_wt)

		timestr = time.strftime("%Y_%m_%d_%H%M%S")
		fp = "pred_output_topN_" + str(self.topN) +"_" +timestr+".tsv"
		fp = res_dir + "/" + fp
		self.savePredictResult2File(self.fn_test,self.Y_pred,fp)	

	def trainTotal(self):

		td = TrainData()

		self.x_wt, self.Ylabels,self.fn_train = td.train_model(self.train_dirname, self.w2v_model_path,self.size)


## Print results 
	def printAnalysis(self,true_pred,y_pred1):

		print "########## Analysing the Model result ##########################"


		math_corr = matthews_corrcoef( true_pred,y_pred1)
		roc_auc = roc_auc_score( true_pred,y_pred1)

		print(classification_report( true_pred,y_pred1))
		print("Matthews correlation :" + str(matthews_corrcoef( true_pred,y_pred1)))
		print("ROC AUC score :" + str(roc_auc_score( true_pred,y_pred1)))


	# get data in a tuple form
	def getAnalysis(self,true_pred,y_pred1):

		precision, recall, fscore, support = score(true_pred,y_pred1)
		return matthews_corrcoef( true_pred,y_pred1),roc_auc_score( true_pred,y_pred1),precision[0],precision[1],recall[0],recall[1],fscore[0],fscore[1],support[0],support[1]

	# print result of start function
	def printResult(self):
		
		self.printAnalysis(self.Y_test,self.Y_pred)

	def getResult(self):

		return self.result

	# write result to file
	def saveResult2file(self,fname):

		fname = res_dir + "/" + fname
		input_filename=open(fname, "wb")
		input_filename.write(("filname\tlabelled\tpredicted\n"))

		for i in range(len(self.Y_test)):
			input_filename.write((self.fn_test[i]+"\t"+str(self.Y_test[i])+"\t"+str(self.Y_pred[i])+"\n"))
		input_filename.close()

	# write predict data result to file
	def savePredictResult2File(self,fn_test,Y_pred,fname):

		fname = res_dir + "/" + fname
		input_filename=open(fname, "wb")
		input_filename.write(("filname\tpredicted\n"))
		print Y_pred
		for i in range(len(Y_pred)):
			input_filename.write((fn_test[i]+"\t"+str(self.Y_pred[i])+"\n"))
		input_filename.close()
		

	# Save N- fold result to file		
	def saveNFoldResult2file(self,y_test,y_pred,fn_test,fname):
		
		fname = res_dir + "/" + fname
		input_filename=open(fname, "wb")
		input_filename.write(("filname\tlabelled\tpredicted\n"))

		for i in range(len(y_test)):
			input_filename.write((fn_test[i]+"\t"+str(y_test[i])+"\t"+str(y_pred[i])+"\n"))
		input_filename.close()



	# calling N-Fold test , inter_N is for number of Test, split defines divide split % in test and (100- split)% in train data
	def NFoldTest(self,total_dirname, iter_N=5,split =0.30,random_state=0):
	
		td = TrainData()
		x_total, y_total,fn_total = td.train_model(total_dirname, self.w2v_model_path,self.size)

		kf_total = cross_validation.ShuffleSplit(len(x_total), n_iter=iter_N, test_size=split, random_state=random_state)
		x_tot_np = Math.array(x_total)
		y_tot_np = Math.array(y_total)


		j =0
		fn_test = []
		timestr = time.strftime("%Y_%m_%d_%H%M%S")
		fnRes = "Output_srl_" + str(self.size)+"_split_"+str(split*100) + "_N_Fold_" + timestr +".tsv"
		fnRes = res_dir + "/" + fnRes
		fp = open(fnRes,"wb")
		fp.write("matthews_corrcoef\troc_auc_score\tprecision[0]\tprecision[1]\trecall[0]\trecall[1]\tfscore[0]\tfscore[1]\tsupport_0\tsupport_1\n")

		for train, test in kf_total:
			j += 1
			print "Case:: " + str(j) +"\n\n"
			lgr = LogisticRegression(C=1.0).fit(x_tot_np[train],y_tot_np[train])
			y_pred = lgr.predict(x_tot_np[test])
			y_test = y_tot_np[test]
			k = 0
			fn_test = []
			for i in test:
				fn_test.append(fn_total[i])
				k += 1

		#	fn = "Output_topN_" + str(j) +"_"+timestr + ".tsv"
			print "writing data to file"
			#self.saveNFoldResult2file(y_test,y_pred,fn_test,fn)
			#self.printAnalysis(y_test,y_pred)
			matthews_corrcoef, roc_auc_score, precision_0 , precision_1 , recall_0 , recall_1 , fscore_0 , fscore_1 , support_0, support_1 = self.getAnalysis(y_test,y_pred)
			Resultstr = str(matthews_corrcoef)+"\t"+str(roc_auc_score)+"\t"+str(precision_0)+"\t"+str(precision_1)+"\t"+str(recall_0)+"\t"+str(recall_1)+"\t"+str(fscore_0)+"\t"+str(fscore_1)+"\t"+str(support_0)+"\t"+str(support_1)+"\n"
			fp.write(Resultstr)



if __name__ == '__main__': 
	train_dirname = '/home/viswanath/workspace/test_resume/train'
	test_dirname = '/home/viswanath/workspace/test_resume/test'
	w2v_model_path = '/home/viswanath/workspace/code_garage/conver2txt/model/w2v_model_100v3.mod'
	size = 100

	gsl = genSRLVec(train_dirname,test_dirname,w2v_model_path,size)
	gsl.trainTotal()
#	gsl.printResult()
#	gsl.saveResult2file("srl_result_v5.tsv")
	gsl.NFoldTest()
#	print gsl.getResult()


