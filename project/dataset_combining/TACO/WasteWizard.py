import requests

class WasteWizard:
    recollect_url = ''
    score_threshold = 0.1
    
    def __init__(self, url='http://api.recollect.net/api/areas/Sacramento'):
        self.recollect_url = url

    def searchTerm(self,query):
        '''Uses fuzzySearch but only returns the top result
        returns title'''
        results = self.fuzzySearch(query)
        # If there's no results, or the results aren't confident, return None
        if len(results) == 0:
            return None
        if float(results[0]['score']) < self.score_threshold:
            return None
        else:
            return results[0]['title']

    def searchId(self, query):
        '''Uses fuzzySearch but only returns the top result
        returns id'''
        results = self.fuzzySearch(query)
        # If there's no results, or the results aren't confident, return None
        if len(results) == 0:
            return None
        if float(results[0]['score']) < self.score_threshold:
            return None
        else:
            return results[0]['id']

    def fuzzySearch(self, query, scores=False):
        '''Searches the waste wizard and returns a list of matches
        serachResult = [{title: plastic, id: 114556, score: 5.6},
                        {title: plastic bottle, id: 45244, score: 4.5}]'''
        url = self.recollect_url+'/services/waste/pages?suggest='+query+'&type=material&set=default&include_links=true&locale=en-US&accept_list=true&_=1666888380289'
        response = self.recollect_api(url)
        searchResults = []
        for result in response.json():
            searchResults.append({
                'title': result['title'],
                'id': result['id'],
                'score': result['score']
            })
        return searchResults

    def getBestOption(self, id):
        '''Searches by id and returns the returns how to dispose of it'''
        url = 'https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/'+id+'.json'
        response = self.recollect_api(url)
        for section in response.json()['sections']:
            try:
                if section['title'].lower() == 'best option':
                    return section['rows'][0]['value']
            except:
                pass
        return None

    def getSpecialInstructions(self, id):
        '''Doesn't work for every entry, but will return HTML instructions for special items'''
        url = 'https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/'+id+'.json'
        response = self.recollect_api(url)
        for section in response.json()['sections']:
            try:
                if section['title'].lower() == 'special instructions':
                    return section['rows'][0]['value']
            except:
                pass
        return None

    def listBestOptions(self):
        '''Lists all possible trash/recycling/compost/hazard categries'''

    def listAllCategories(self):
        return ["trash", "reyclcing"]

    def recollect_api(self, url):
        proxies = {
            'http': 'http://proxy-chain.intel.com:911'
        }
        response = requests.get(url, proxies=proxies)
        if response.status_code == 200:
            return response
        else:
            print("Failed response from API")

def main():
    ww = WasteWizard()
    print(ww.searchId("paper"))
    print(ww.searchTerm("person"))
    print(ww.fuzzySearch("paper"))
    print(ww.getBestOption(ww.fuzzySearch("plastic")[0]['id']))

if __name__ == "__main__":
    main()