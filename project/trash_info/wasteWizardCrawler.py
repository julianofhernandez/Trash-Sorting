# Material List:

# API request
import requests


searchList = [
        "garbage bag", "face masks", "diapers","Styrofoam","pet waste","cooking oil","clam shell trays", "deli food containers", "Ziplock bags", "inside cereal box plastic", "bubble wrap", "clear plastic wrap",
        'Clear glass','Green glass','Brown glass','Blue glass','Aluminum and tin cans','Aluminum trays and foil rinsed','Empty aerosol cans','Pots, pans and utensils','Lids from jars','Soda bottles', 
        'milk jug', 'shampoo bottles''Buckets, pails and crates','Cardboard','Cereal boxes','Paper bags','Paper packaging','Junk mail','Books','Office paper',
        'fruit', 'vegetable', 'greasy paper container','paper towels and napkins','coffee filters and tea bags','paper takeout with no wax or plastic lining',
        'electronic waste','people','paint','batteries','chemicals','Motor oil','fluorescent bulbs','medical sharps','clothing','fuel tanks'
    ]

def main():
    for searchItem in searchList:
        print(searchItem + ": " + searchForBin(searchItem))
        #print("     Special instuctions: " + searchForSpecialInstructions(searchItem))


# Get suggestion list
# https://api.recollect.net/api/areas/Sacramento/services/waste/pages?suggest=paper&type=material&set=default&include_links=true&locale=en-US&accept_list=true&_=1666888380289

# Use ID from suggestion list to search here 
# https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/473087.json

def get_id(searchTerm):
    url = 'http://api.recollect.net/api/areas/Sacramento/services/waste/pages?suggest='+searchTerm+'&type=material&set=default&include_links=true&locale=en-US&accept_list=true&_=1666888380289'
    response = recollect_api(url)
    if len(response.json()) == 0:
        print('no id found for ' + searchTerm)
        return None
    else:
        waste = response.json()[0]
        # print('Searching for: '+searchTerm+' but found '+waste['title']+'\n')
        return waste['id']

def searchForBin(searchTerm):
    id = get_id(searchTerm)
    if (id is not None):
        url = 'https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/'+id+'.json'
        response = recollect_api(url)
        for section in response.json()['sections']:
            try:
                if section['title'].lower() == 'best option':
                    return section['rows'][0]['value']
            except:
                pass
        print("ID not found for:"+searchTerm)

def searchForSpecialInstructions(searchTerm):
    id = get_id(searchTerm)
    if (id is not None):
        url = 'https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/'+id+'.json'
        response = recollect_api(url)
        for section in response.json()['sections']:
            try:
                if section['title'].lower() == 'special instructions':
                    return section['rows'][0]['value']
            except:
                pass
        print("special instructions not found for:"+searchTerm)

def recollect_api(url):
    proxies = {
    'http': 'http://proxy-chain.intel.com:911'
    }
    response = requests.get(url, proxies=proxies)
    if response.status_code == 200:
        return response
    else:
        print("Failed response from API")

if __name__ == "__main__":
    main()