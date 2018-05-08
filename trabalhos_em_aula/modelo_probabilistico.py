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

def calc_b(freq, estm_len, avg_doclen, K1, B):
	return ((K1 + 1)*freq) / (K1*( (1-B) + B*(estm_len/avg_doclen) ) +freq)

def bm25(M, word_vec_matrix, word_vec_q, K, b):

	num_docs=len(word_vec_matrix)
	num_docs_term=[]

	valid_terms=[]

	for row, term in zip(np.transpose(word_vec_matrix), word_vec_q):
		if term:
			num_docs_term.append(len(np.nonzero(row)[0]))
			valid_terms.append(row)

	valid_terms_per_doc=np.transpose(valid_terms)


	words_per_doc=[]

	for line in word_vec_matrix:
		words=0
		for freq in line:
			words+=freq
		words_per_doc.append(words)
	avg_doclen=np.mean(words_per_doc)

	B = [
		[calc_b(freq, words_per_doc[idx], avg_doclen, K, b) for freq in line]
		for idx, line in enumerate(valid_terms_per_doc)
	]

	rank=[]
	for idx, line in enumerate(B):
		soma=0
		for idx2, b in enumerate(line):
			soma+=(b*math.log(((num_docs-num_docs_term[idx2]+0.5)/(num_docs_term[idx2]+0.5)),2))
		rank.append((M[idx], soma))

	rank.sort(reverse=True, key=lambda x: x[1])
	return rank



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

print "dictionary"
print dictionary
print " "
word_vec_question=create_wordvec(q, dictionary)

print "rank :"

var = bm25(M, word_vec_matrix, word_vec_question, 1, 0.75)
print var
f = open('testfile.txt','w')
for v in var:
	if v[1] > 0:
		f.write(str(M.index(v[0]) + 1) + '\n')
f.close()
