# -*- coding: utf-8 -*-
import math
import numpy as np

def intersect(a,b):
    lista = []
    for a_ in a:
        if a_ in b and a_ not in lista:
            lista.append(a_)
    return lista

def calc_Dr(recuperados, R):
    return intersect(recuperados,R)

def calc_Dn(recuperados, R):
    Dn=[]
    inter=intersect(recuperados,R)
    for doc in recuperados:
        if doc not in inter:
            Dn.append(doc)
    return Dn

def calc_qm(q, Dr, Dn, alpha=1.0, beta=0.75, gama=0.15):
    qm=[]
    for i in range(0,len(q)):
        qm.append(alpha*q[i] + (beta/len(Dr))*sum([doc[i] for doc in Dr]) - (gama/len(Dn))*sum([doc[i] for doc in Dn]))
    return qm


#SAME OLD MODELO VETORIAL

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

def tf(word_vec_matrix):
	tf_matrix=[]
	for line in word_vec_matrix:
		tf_matrix.append(tf_line(line))
	return tf_matrix

def tf_line(line):
    line_tf=[]
    for freq in line:
        if (freq>0):
            line_tf.append(1+math.log(freq,2))
        else:
            line_tf.append(0)
    return line_tf


def idf(word_vec_matrix, number_of_files):
	idf=[]
	for row in np.transpose(word_vec_matrix):
			idf.append(math.log(float(number_of_files)/len(np.nonzero(row)[0]),2))
	print "\nidf: ", len(idf)
	print idf
	print "\n"
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

def normalize(vector):
	sum=0
	for element in vector:
		sum+=element**2
	return np.sqrt(sum)

def create_rank(sentences_matriz, query_vector, norm_sentences, norm_query):
	rank=[]
	for idx, line in enumerate(sentences_matriz):
		rank.append(np.dot(line, query_vector)/(norm_sentences[idx]*norm_query))
	return rank

R = ['1','2']
recuperados=open('vet-rec.txt', 'r').read().split()

M=['O pea e o caval são pec de xadrez. O caval é o melhor do jog.',
'A jog envolv a torr, o pea e o rei.',
'O pea lac o boi',
'Caval de rodei!',
'Polic o jog no xadrez.']

stopwords=['','a', 'o', 'e', 'é', 'de', 'do', 'no','são']

separators=[' ',',','.','!','?']

q='xadrez caval torr pea'
q=q.split(' ')

print 'arquivos:'
for i in M:
	print i

print '\n\nconsulta: '
print q
print ' '

#AULA 14
Dr=calc_Dr(recuperados, R)
Dn=calc_Dn(recuperados, R)

print "Dr = ", Dr
print "Dn = ", Dn
print " "


tokenized_matrix=tokenize(M, separators, stopwords)
dictionary=create_dictionary(tokenized_matrix)

print "dictionary: "+str(len(dictionary))
print dictionary
print "\n"

word_vec_matrix=[]
for sentence in tokenized_matrix:
	word_vec_matrix.append(create_wordvec(sentence, dictionary))

print "matriz de incidencia: "
print word_vec_matrix
print "\n"

tf_idf_sentences=tf_idf(tf(word_vec_matrix), idf(word_vec_matrix, len(M)))

print "tf-idf of the sentences:"
for e in tf_idf_sentences:
	print e
print '\n\n'

word_vec_question=create_wordvec(q, dictionary)

print "original q = ", word_vec_question

#AULA 14
Dr_word_vec=[]
Dn_word_vec=[]

for doc in Dr:
    Dr_word_vec.append(word_vec_matrix[int(doc)-1])
for doc in Dn:
    Dn_word_vec.append(word_vec_matrix[int(doc)-1])

qm=calc_qm(word_vec_question, Dr_word_vec, Dn_word_vec)

print "qm = ", qm
print " "


tf_idf_question=tf_idf_line(tf_line(qm), idf_line(qm, len(M)))

print "tf-idf of the qm:"
print tf_idf_question
print '\n\n'


#aula 16/4
norm_matriz_senteces=[]
for line in tf_idf_sentences:
		norm_matriz_senteces.append(normalize(line))

norm_vec_question=normalize(tf_idf_question)

rank=create_rank(tf_idf_sentences, tf_idf_question, norm_matriz_senteces, norm_vec_question)

ordered_documents=[(rank[x],M[x]) for x in range(len(rank))]
result=sorted(ordered_documents)

print "documents ordened by modified query qm:"
result.reverse()
for i in result:
	print i
