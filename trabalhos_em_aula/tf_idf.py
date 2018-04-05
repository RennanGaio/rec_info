# -*- coding: utf-8 -*-
import math
import numpy as np

def tokenize(sentence_list, separators, stopwords):
	tokenized=[]

	for sentence in sentence_list:
		temp=[]
		sentence=sentence.lower()
		#tokenize sentence
		for sep in separators:
			sentence=sentence.replace(sep, ' ')
		sentence=sentence.split(' ')
		#take off stop words
		for word in sentence:
			if word not in stopwords:
				temp.append(word)
		tokenized.append(temp)
	return tokenized

def create_dictionary(tokenized_matrix):
	dictionary=[]
	for sentence in tokenized_matrix:
		for word in sentence:
			if word not in dictionary:
				dictionary.append(word)
	return dictionary

def create_wordvec(tokenized_sentence, dictionary):
	word_vec=[]
	for word in dictionary:
		word_vec.append(tokenized_sentence.count(word))
	return word_vec

def tf(word_vec_matriz):
	tf_matrix=[]
	for line in word_vec_matriz:
		tf_matrix.append(tf_line(line))
	return tf_matrix

def tf_line(line):
	line_tf=[]
	for freq in line:
		if (freq):
			line_tf.append(1+math.log(freq,2))
		else:
			line_tf.append(0)
	return line_tf

def idf(word_vec_matriz, number_of_files):
	idf=[]
	for row in np.transpose(word_vec_matriz):
		idf.append(math.log(number_of_files/len(np.nonzero(row)[0]),2))
	return idf

def idf_line(word_vec, number_of_files):
	idf_line=[]
	for row in word_vec:
		if row:
			idf_line.append(math.log(number_of_files/len(np.nonzero(row)[0]),2))
		else:
			idf_line.append(0)
	return idf_line


def tf_idf(tf, idf):
	tf_idf=[]
	for line in tf:
		tf_idf.append(tf_idf_line(line, idf))
	return tf_idf

def tf_idf_line(tf_line, idf):
	tf_idf_vec=[]
	for idx, element in enumerate(tf_line):
		tf_idf_vec.append(element*idf[idx])
	return tf_idf_vec

M=['O peã e o caval são pec de xadrez. O caval é o melhor do jog.',
'A jog envolv a torr, o peã e o rei.',
'O peã lac o boi',
'Caval de rodei!',
'Polic o jog no xadrez.']

stopwords=['','a', 'o', 'e', 'é', 'de', 'do', 'no','são']

separators=[' ',',','.','!','?']

q='xadrez peã caval torr'
q=q.split(' ')

print 'arquivos:'
for i in M:
	print i

print '\n\nconsulta: '
print q
print ' '


tokenized_matrix=tokenize(M, separators, stopwords)
dictionary=create_dictionary(tokenized_matrix)

word_vec_matrix=[]
for sentence in tokenized_matrix:
	word_vec_matrix.append(create_wordvec(sentence, dictionary))

tf_idf_sentences=tf_idf(tf(word_vec_matrix), idf(word_vec_matrix, len(M)))

print "tf-idf of the sentences:"
print tf_idf_sentences
print '\n\n'

word_vec_question=create_wordvec(q, dictionary)
tf_idf_question=tf_idf_line(tf_line(word_vec_question), idf_line(word_vec_question, len(M)))

print "tf-idf of the question:"
print tf_idf_question
