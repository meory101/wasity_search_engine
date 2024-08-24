import requests

response =  requests.get('http://127.0.0.1:8000/api/getProducts')
print( response.content)


response =  requests.post('http://10.2.0.2:5000/api/getMostReleventProducts',data={'searchQuery':'skskks'})
print( response.content)


