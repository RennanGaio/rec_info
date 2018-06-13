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

def measure_by_relevant(R, recuperados):
    lista = []
    table = [["recall", "precision"]]
    for r_ in recuperados:
        lista.append(r_)
        if r_ in R:
            table.append([recall (R, lista), precision (R, lista)])
    return table

def precisao_interpolada(table):
    table.pop(0)
    tabela_interpolada = []
    for i in range(0, 11):
        max_value = 0
        for j in table:
            if (float(i)/10) <= j[0] and max_value < j[1]:
                max_value=j[1]
        tabela_interpolada.append([float(i)/10, max_value])
    return tabela_interpolada

def MAP(table, R):
    return float(sum([x[1] for x in table]))/len(R)

if __name__ == '__main__':
    R = ['1','2']
    recuperados=open('bm25-rec.txt', 'r').read().split()
    #recuperados=open('vet-rec.txt', 'r').read().split()

    print (recuperados)
    print ("\n")
    print ("recall: "+ str(recall(R, recuperados)))
    print ("\n")
    print ("precision: "+ str(precision(R, recuperados)))
    print ("\n")
    print ("f1 - ", f_measure(1, len(recuperados), R, recuperados))
    print ("\n")
    table = measure_by_relevant(R, recuperados)

    print ("revocacao e precisao a cada documento recuperado: \n", table)

    print ("\nprecisao interpolada: ", precisao_interpolada(table))

    print ("\nMAP: ", MAP(table, R))
