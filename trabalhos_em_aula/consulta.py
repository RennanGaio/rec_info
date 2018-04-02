# -*- coding: utf-8 -*-

M=['O peã e o caval são pec de xadrez. O caval é o melhor do jog.',
'A jog envolv a torr, o peã e o rei.',
'O peã lac o boi',
'Caval de rodei!',
'Polic o jog no xadrez.']

stopwords=['','a', 'o', 'e', 'é', 'de', 'do', 'no','são']

separadores=[' ',',','.','!','?']

q='xadrez peã caval torr'

print 'arquivos:'
for i in M:
	print i

print '\n\nconsulta: '
print q
print ' '

q=q.split(' ')

#tokenizacao dos arquivos
M2=[]
for i in M:
	i=i.lower()
	for sep in separadores:
		i=i.replace(sep, ' ')
	i=i.split(' ')
	M2.append(i)

word_list=[]
M3=[]

#retirada de stop words
for i in M2:
	temp=[]
	for token in i:
		if token not in stopwords:
			temp.append(token)
	M3.append(temp)
		
#criacao de word vec
for i in M3:
	for token in i:
		if token not in word_list:
			word_list.append(token)

M_inc=[]

#word vec para cada arquivo
for i in M3:
	temp=[]
	for token in word_list:
		temp.append(i.count(token))
	M_inc.append(temp)

#word vec para a consulta
q_inc=[]
for token in word_list:
	q_inc.append(q.count(token))

#separacao dos index que sao relevantes para a consulta
index_list=[]
for idx, element in enumerate(q_inc):
	if element:
		index_list.append(idx)


func=str(raw_input('digite a funcao que deseja executar ["and" ou "or"]: '))

#consulta
for idx,i in enumerate(M_inc):
	is_in_question=bool(i[index_list[0]])
	for j in index_list[1:]:
		if func=='and':
			is_in_question=is_in_question & bool(i[j])
		if func=='or':
			is_in_question=is_in_question | bool(i[j])
	if is_in_question:
		print M[idx]

