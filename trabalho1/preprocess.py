from bs4 import BeautifulSoup
from os import listdir


def preprocess(text):
    separators=separators=[' ',',','.','!','?',';','[',']','{','}','(',')',':','<','>']
    stopwords=open('stopwords.txt').read()
    temp=[]
    text=text.lower()
	#tokenize sentence
    for sep in separators:
        text=text.replace(sep, ' ')
    text=text.split(' ')
	#take off stop words
    for word in text:
        if word not in stopwords:
            temp.append(word)
    return " ".join(temp)


files = [f for f in listdir('colecao_teste')]

for ofile in files:
    f = open('colecao_teste/' + ofile, encoding="latin1")
    sgml = f.read()
    soup = BeautifulSoup(sgml,"lxml")

    sfile = open('colecao_stem/'+ ofile, 'a')
    for doc in soup.find_all('doc'):
        text = preprocess(doc.find('text').get_text())
        docno = doc.find('docno').get_text()

        sfile.write("<DOC>\n\
<DOCNO>"+docno+"</DOCNO>\n\
<TEXT>"+text+"</TEXT>\n\
</DOC>\n")
