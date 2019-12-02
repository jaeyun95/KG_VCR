# extract Fact Using ConceptNet

import requests

#give one word example
#obj = requests.get('http://api.conceptnet.io/c/en/person?offset=0&limit=1000').json()
#print(len(obj['edges']))

#give two word example
#response = requests.get('http://api.conceptnet.io/query?node=/c/en/person&other=/c/en/bite')

#give one word and relation example
#response = requests.get('http://api.conceptnet.io/query?node=/c/en/school&rel=/r/CapableOf')
#obj = response.json()

#give one word
def extractFact(word):
    offset = 0
    jsonList = []
    while True:
       obj = requests.get('http://api.conceptnet.io/c/en/'+word+'?offset='+str(offset)+'&limit=1000').json()
       offset = int(offset) + 1000
       #print(len(obj['edges']))
       jsonList.append(obj['edges'])
       if(len(obj['edges'])<1000):
           break
    return jsonList

#give two word
def extractFactTwo(word1,word2):
    jsonList = []
    response = requests.get('http://api.conceptnet.io/query?node=/c/en/'+word1+'&rel=/r/RelatedTo')
    obj = response.json()
    jsonList.append(obj['edges'])
    return jsonList
	