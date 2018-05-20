import subprocess

filename="rec_info_stop_trab1"
consulta=1
queries=open("query.txt").read().split("\n")
rank=open("rank_stop.txt", "w")
queries.pop(-1)
dupla= "Rennan_Alex"
stopwords_file= "stopwords.txt"

for query in queries:
    command="zet -n 100 -f " + filename + " --query-stop "+ stopwords_file+ " " + query
    zet_output=subprocess.check_output([command], shell=True)
    zet_list=zet_output.split("\n")

    #remove trash from zet output in subprocess call
    del zet_list[-4:]

    for element in zet_list:
        e=element.split(" ")
        element_rank=e[0].replace(".","")
        doc_id=e[1]
        element_score=e[3].replace(",","")
        line_string=str(consulta)+" Q0 "+doc_id+" "+element_rank+" "+element_score+" "+dupla+"\n"
        rank.write(line_string)
    consulta+=1
