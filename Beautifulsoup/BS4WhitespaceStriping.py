from bs4 import BeautifulSoup
import requests

url = "https://www.example.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
data = soup.get_text(strip=True)
print("Data:", data)
