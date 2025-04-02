import json
import os

import pandas as pd

prev_directory = os.path.dirname(os.getcwd())

with open(rf'{prev_directory}/resources/labourlawterminologyv2.jsonld', encoding='utf-8') as file:
    data = json.load(file)

synonyms_list =[]
broader_dict = {}

i = 0
aux = data[0]['@graph']
#se recorre el json en busca de las palabras y sus términos relacionados en español
for elements in aux:
    word = ''
    synonyms = []
    lt = []
    keys = elements.keys()
    for key in keys:
        if key.endswith('prefLabel'):
            for terms in elements[key]:
                if terms['@language'] == 'es':
                    word = terms['@value'].lower()
        if key.endswith('altLabel'):
            for terms in elements[key]:
                if terms['@language'] == 'es':
                    synonyms.append(terms['@value'].lower())
        if key.endswith('broader'):
            for broaders in elements[key]:
                lt.append(broaders)

    #si hay una palabra con sinónimos en español se almacena en un fichero csv
    if word != '' or len(synonyms) != 0:
        if len(synonyms) != 0 and word == '':
            word = synonyms[0]
            del synonyms[0]
        result_word = {
            "palabra": word,
            "sinonimos": synonyms,
            "broader": lt
        }
        synonyms_list.append(result_word)
        broader_dict[elements['@id']] = [word] + synonyms

df = pd.DataFrame(synonyms_list)
synonyms_list.clear()
for i in range(len(df)):
    aux = df.iloc[i]
    word = aux['palabra']
    new_synonyms = aux['sinonimos']
    if len(aux['broader']) != 0:
        for broader in aux['broader']:
            #si el broader asociado tiene términos en español los añadimos a sinónimos
            if broader['@id'] in broader_dict.keys():
                new_synonyms += broader_dict[broader['@id']]
    if len(new_synonyms) != 0:
        sol = {
            "palabra": word,
            "sinonimos": new_synonyms
        }
        synonyms_list.append(sol)
df2 = pd.DataFrame(synonyms_list)
csv_path = fr"{prev_directory}/resources/synonyms_list.csv"
df2.to_csv(csv_path)