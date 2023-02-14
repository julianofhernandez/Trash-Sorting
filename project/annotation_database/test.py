import requests

path = 'http://127.0.0.1:5000/'


'''
file = {'image': open('test.jpg','rb')}
create_data = {'key': 'secretkey', 'name':"test1.png",
               'annotations':'{blablalba}', 'num_annotation':999, 'metadata':'{"foo"}'}

r = requests.post(path + 'create/entry', data = create_data, files = file)

print(r.text, r.status_code)




file = {'image': open('test2.jpg','rb')}
create_data = {'key': 'secretkey', 'name':"test2.png",
               'annotations':'{}', 'num_annotation':0, 'metadata':'{}'}

r = requests.post(path + 'create/entry', data = create_data, files = file)

print(r.text, r.status_code)

r = requests.get(path + '/read/annotation/max')

print(r.text , r.status_code)


#r = requests.get(path + '/read/count/foo')

#print(r.text , r.status_code)
'''

r = requests.get(path + '/read/entry/data/1')

print(r.text , r.status_code)
