import requests
from bs4 import BeautifulSoup
import lxml


def get_yt_tags(url):
    url = url.group()
    words = ''
    if 'youtube' in url:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            v_tags = soup.find_all('meta', property='og:video:tag')
            tags = [tag['content'] for tag in v_tags]
            genre = soup.find('meta', itemprop='genre')
            if genre:
                tags.append(genre['content'])
            words = ' '.join(tags)
    return words
