import requests
import json
import pandas as pd
import os

def check_exist_lang(lang,r, wiki_id):
    #checks if exists a wikipedia link of a given language of a given subject
    wiki = lang+"wiki"
    return wiki in r.json()['entities'][wiki_id]['sitelinks']

def find_files(source_L="en", target_L="pt"):
    #On this function we gonna find the missing articles of a target language given a source languange.
    #This could be a general function, but for this project we gonna reduce our search, looking only for english files that misses
    #portuguese files on wikipedia.
    itens_ID_exists=[]
    itens_ID_dont_exists=[]
    error_cont=0

    #for j_file in ["queryP4466.json", "queryP118.json"]:
    for j_file in ["queryP4466.json"]:

        with open(j_file) as f:
            wiki = json.load(f)

        wikidata_concepts=list(set([item['item'].split("/")[-1] for item in wiki if item['item'].split("/")[-1][0] == "Q"]))

        for wiki_id in wikidata_concepts:
            #vars for names of the wikipedia files in pt and eng
            eng_name=""
            pt_name=""
            #colect data that we need to identify if this subject already have pt information
            command = "api.php?action=wbgetentities&ids="+wiki_id+"&redirects=no&format=json"
            try:
                r = requests.get('https://www.wikidata.org/w/'+command)
            except Exception as e:
                r = ""
                print "error in url"
                error_cont+=1

            if r!="" and r.json()['success']:
                if (check_exist_lang(source_L, r, wiki_id)):
                    eng_name=r.json()['entities'][wiki_id]['sitelinks']['enwiki']['title'].replace(" ","_")
                    if (check_exist_lang(target_L, r, wiki_id)):
                        eng_name=r.json()['entities'][wiki_id]['sitelinks']['ptwiki']['title'].replace(" ","_")
                        itens_ID_exists.append([wiki_id, eng_name, pt_name])
                    else:
                        itens_ID_dont_exists.append([wiki_id, eng_name, pt_name])
                #print (r.json()['entities'])

    #save all those ids in a file for backup if necessary
    # with open("wiki_br.txt", "w") as f:
    #     f.write(itens_ID_exists)
    # with open("wiki_no_br.txt", "w") as f:
    #     f.write(itens_ID_dont_exists)

    print "total errors: "+str(error_cont)
    return itens_ID_exists, itens_ID_dont_exists

def search_index_views(iten, data,has_pt):
    eng_views=data[data['wiki_name'] == iten[1]]['views'].unique()[0]
    if has_pt:
        pt_views=data[data['wiki_name'] == iten[2]]['views'].unique()[0]
        return eng_views, pt_views
    return eng_views

def associate_page_views(itens_ID_exists, itens_ID_dont_exists):
    page_views_file = open("page_views.csv", "a")
    missing_rank_file = open("missing_rank.csv", "a")

    files = glob.glob('pageview-11-05-2015/*')
    #second column convert to datetimeindex
    dfs = [pd.read_csv(fp, sep=" ",index_col=[1], header=None) for fp in files]
    data = pd.concat(dfs).sort_index()
    data.columns = ["lang", "wiki_name", "views", "size"]

    for i in itens_ID_exists:
        eng_views, pt_views=search_index_views(i, data,1)
        line=str(i[0])+" "+str(eng_views)+" "+str(pt_views)
        page_views_file.append(line)
    for i in itens_ID_dont_exists:
        eng_views=search_index_views(i, data,0)
        line=str(i[0])+" "+str(eng_views)+" NULL"
        missing_rank_file.append(line)



def rank_articles():


#def match_editors():

if __name__ == '__main__':
    itens_ID_exists, itens_ID_dont_exists = find_files()
    associate_page_views(itens_ID_exists, itens_ID_dont_exists)
    rank_articles()
