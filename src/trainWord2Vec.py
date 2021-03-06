import sys, re, os, ast
import gensim, logging

from extSent import Sentences

from configobj import ConfigObj

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', format=logging.INFO)



#print top N similar words
def print_similars(model, item_list, N):
	print 'Top ' + str(N) + ' Items similar to:' + item_list
	for each in model.most_similar(item_list, [], N):
		print each

# test function
def test(model):

	positive = ['java']
	negative = []

	print "########## POSITIVE = {0} ##### NEGATIVE = {1}".format(positive, negative)
	print model.most_similar_cosmul(positive,negative)

# function used for training word2vec from files in input_data_dir and write word2vec model in out_model_file using parameter 
# size : size of vector
# window: subset of word used for training data
# negative: neagtive sample to be used for training
# sample: sampling parameter
# min_count: minimum number of occurences in the set
# workers: No. of threads to be used
# iterations: No. of iterations to be done for training
def train(model_name, is_phrased, input_data_dir, size, window, negative, sample, min_count, workers, iterations, out_model_file):

	
	data = Sentences(input_data_dir)

	model = gensim.models.Word2Vec(size=size, window=window, negative=negative, sample=sample, min_count=min_count, workers=workers, iter=iterations)
	model.build_vocab(data)
	
	T = model.train(data)

	out_model_file = out_model_file +"_size"+str(size)+"_window"+ str(window)+"_negative"+ str(negative)+"_sample"+ str(sample)+"_minCount"+ str(min_count)+"_iter"+  str(iterations)+".mod"

	model.save(out_model_file)
	print 'Model: ' + model_name + ' saved to disk in: [' + out_model_file + '.'

	print 'Model trained: ' + str(model)
	print 'Model vocab count ' + str(len(model.vocab))
	
	return model
	
def genWord2Vec(model_name, is_phrased, input_data_dir, size, window, negative, sample, min_count, workers, iterations, out_model_file):
	
	data = Sentences(input_data_dir)

	model = gensim.models.Word2Vec(size=size, window=window, negative=negative, sample=sample, min_count=min_count, workers=workers, iter=iterations)
	model.build_vocab(data)
	
	T = model.train(data)

	out_model_file = out_model_file +"_size"+str(size)+"_window"+ str(window)+"_negative"+ str(negative)+"_sample"+ str(sample)+"_minCount"+ str(min_count)+"_iter"+  str(iterations)+".mod"

	model.save(out_model_file)
	print 'Model: ' + model_name + ' saved to disk in: [' + out_model_file + '.'

	print 'Model trained: ' + str(model)
	print 'Model vocab count ' + str(len(model.vocab))
	
	return model


### print vocab for the word2vec
def print_vocab(model):
	for each in model.vocab:
		print each
	print 'Vocab count: ' + str(len(model.vocab))

def train_model(config):

	model_name = config['word2vec']['model_name']

	print '########### Training word2vec model: ' + model_name

	model = train(model_name,
			config['word2vec']['train']['is_phrased'],
			config['word2vec']['train']['input_file'],
			int(config['word2vec']['train']['size']), 
			int(config['word2vec']['train']['window']),
			int(config['word2vec']['train']['negative']),
			float(config['word2vec']['train']['sample']),
			int(config['word2vec']['train']['min_count']),
			int(config['word2vec']['train']['workers']),
			int(config['word2vec']['train']['iterations']),
			config['word2vec']['train']['out_model_file'])
	return model

def test_model(config):
			
	model_name = config['word2vec']['model_name']

	print '########### Testing word2vec model: ' + model_name

	model = gensim.models.Word2Vec.load(config['word2vec']['test']['model_file'])
#	test(model)

def load_W2Vec(modelPath):
	return gensim.models.Word2Vec.load(modelPath)


def main():
	
	config = ConfigObj('/home/viswanath/workspace/code_garage/conver2txt/config/config_word2vec.ini')
	actions = config['main']['action']
	
	for action in actions.split(','):
		
		if   action == 'train':
			train_model(config)
		elif action == 'test':
			test_model(config)
		elif action == 'phrase':
			phrasing(config)
		else:
			print 'Invalid instruction in config file.'


if __name__ == '__main__': main()

