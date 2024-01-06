from bs4 import BeautifulSoup
import requests
url="https://www.example.com"
response=requests.get(url)
soup=BeautifulSoup(response.text,"html.parser")
print("Beautiful Soup HTML format",soup)
data=soup.get_text()
print("Data :",data)
