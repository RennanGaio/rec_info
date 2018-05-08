def intersect(a,b):
    lista = []
    for a_ in a:
        if a_ in b and a_ not in lista:
            lista.append(a_)
    return lista

def recall (R, recuperados):
    return float(len(intersect(R,recuperados))) / float(len(R))

def precision (R, recuperados):
    return float(len(intersect(R,recuperados))) / float(len(recuperados))

def f_measure(beta, j, R, recuperados):
    return ( (1 + beta**2) * precision(R,recuperados) * recall(R,recuperados)  ) / ((beta**2 * precision(R,recuperados)) + recall(R,recuperados))

if __name__ == '__main__':
    R = ['1','2']
    recuperados=open('testfile.txt', 'r').read().split()

    print recuperados
    print "recall: "+ str(recall(R, recuperados))
    print "precision: "+ str(precision(R, recuperados))

    print ("f1 - ", f_measure(1, len(recuperados), R, recuperados))
