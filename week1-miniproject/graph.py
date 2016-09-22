import requests
from bs4 import BeautifulSoup
from urlparse import urljoin
from datetime import datetime
from ediblepickle import checkpoint
import os
import string
import nltk
import re
import cgi
from lxml import html

cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

baseUrl = "http://www.newyorksocialdiary.com/party-pictures"

def getPageHTML (url) :
	response = requests.get(url)
	return BeautifulSoup(response.text, "html.parser")

def getDates (url) :
	soup = getPageHTML(url)
	dates = soup.select('span.views-field-created .field-content')
	dateTimes = []
	for date in dates :
		dateString = date.get_text()
		dateString = html.fromstring(dateString).text_content()
		dateString = dateString.encode("ascii")
		dateTime = datetime.strptime(dateString, '%A, %B %d, %Y')
		dateTimes.append(dateTime)
	return dateTimes

@checkpoint(key=string.Template('firstParty.csv'), work_dir=cache_dir, refresh=False)
def getFirstParty () :	
	firstParty = False
	page = 0
	while (firstParty == False) :
		dates = getDates(urljoin(baseUrl, '?page=' + str(page)))
		for i, date in enumerate(dates) :
			if (date.year <= 2014) :
				if (date.month < 12) :
					firstParty = date
					break
		if (firstParty == False) :
			page = page + 1
	return firstParty, page, i

@checkpoint(key='urls.csv', work_dir=cache_dir, refresh=False)
def getUrls (firstPage, i) :
	soup = getPageHTML(urljoin(baseUrl, '?page=' + str(firstPage)))
	numPages = soup.select('.pager__item--last a')[0]['href'].split('=')[1]
	urls = []
	for page in range(firstPage, int(numPages) + 1) :
		if (page == firstPage) :
			links = soup.select('.views-field-title .field-content a')[i:]
		else :
			soup = getPageHTML(urljoin(baseUrl, '?page=' + str(page)))
			links = soup.select('.views-field-title .field-content a')
		for link in links :
			href = link['href']
			href = html.fromstring(href).text_content()
			href = href.encode("ascii")
			urls.append(href)
	return urls

@checkpoint(key=lambda args, kwargs: 'captions' + '.p', work_dir=cache_dir)
def getCaptions(urls) :
	captions = []
	for url in urls :
		soup = getPageHTML((urljoin(baseUrl, url)))
		pageCaptions = soup.select('.photocaption')
		if (pageCaptions == []) :
			pageCaptions = soup.select('font[size="1"]')
		for caption in pageCaptions :
			captionText = caption.get_text()
			captionText = html.fromstring(captionText).text_content()
			captionText = captionText.encode("ascii")
			if (len(captionText) <= 250) :
				captions.append(captionText)
	return captions

@checkpoint(key=lambda args, kwargs: 'captions2' + '.p', work_dir=cache_dir, refresh=False)
def filterSentences(captions) :
	filteredCaptions = []
	for caption in captions :
		verb = False
		text = nltk.word_tokenize(caption)
		taggedText = nltk.pos_tag(text)
		for word, pos in enumerate(taggedText) :
			if pos[1][0] == 'V':
				verb = True
				pass
		if verb == False :
			filteredCaptions.append(caption)
	return filteredCaptions

@checkpoint(key=lambda args, kwargs: 'captions3' + '.p', work_dir=cache_dir, refresh=False)	
def splitComma (captions) :
	filteredCaptions = []
	for caption in captions :
		caption = caption.replace("and", ",")
		wordList = caption.split(',')
		filteredCaptions.append(wordList)
	return filteredCaptions

@checkpoint(key=lambda args, kwargs: 'captions4' + '.p', work_dir=cache_dir, refresh=True)	
def removeTitles(captions) :
	filteredCaptions = []
	for caption in captions :
		filteredPersons=[]
		for person in caption :
			person = person.replace("\n", "")
			person = person.strip().lower()
			if(" m.d. " in person) :
				print('m.d. : ' + person)
				person = person.replace(" m.d. ", "")
				filteredPersons.append(person)
			if(" md " in person) :
				person = person.replace(" md ", "")
				filteredPersons.append(person)
			if(" mayor " in person) :
				person = person.replace(" mayor ", "")
				filteredPersons.append(person)
			if(" mrs " in person) :
				person = person.replace(" mrs ", "")
			if(" mr " in person) :
				person = person.replace(" mr ", "")
				filteredPersons.append(person)
			if(" ms " in person) :
				person = person.replace(" ms ", "")
			if(" dr. " in person) :
				person = person.replace(" dr. ", "")
				filteredPersons.append(person)
			else:
				filteredPersons.append(person)
		filteredCaptions.append(filteredPersons)
	return filteredCaptions
	
@checkpoint(key=lambda args, kwargs: 'popularityList' + '.p', work_dir=cache_dir, refresh=False)	
def getPopularity(captions) :
	names = {}
	for caption in captions :
		for person in caption :
			person = person.replace("\n", "")
			person = person.replace("\u", "")
			person = person.strip()
			if person == 'u' :
				pass
			if person in names :
				names[person] = names[person] + 1
			else :
				names[person] = 1
	print type(names)
	return names


firstParty, page, i = getFirstParty()
urls = getUrls(page, i)
captions = getCaptions(urls)
captions = filterSentences(captions)
captions = splitComma(captions)
caption = removeTitles(captions)

for caption in captions :
	print('[')
	for person in caption :
		print(person + ", ")
	print("]")
print(len(captions))
#captions = filterAnds(captions)
#popularityList = getPopularity(captions)
#listname = []  

# for key, value in sorted(popularityList.iteritems(), key=lambda (k,v): (v,k),reverse=True):  
#     diction= {"pop":value, "name":key}  
#     listname.append(diction)
# print listname[0:200]

#130297
#129022
#126928

