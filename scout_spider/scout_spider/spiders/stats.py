# -*- coding: utf-8 -*-
import scrapy
from scout_spider.items import Stats
import re

years = ['2018','2017','2016','2015','2014','2013','2012','2011']

class StatsSpider(scrapy.Spider):
	name = 'stats'
	allowed_domains = ['hockeydb.com']
	start_urls = ['http://www.hockeydb.com/ihdb/draft/nhl{}e.html'.format(year) for year in years]

	def parse(self, response):
		table = response.xpath('//*[@class="sortable autostripe"]//tbody')
		rows = table.xpath('//tr')
		for row in rows:
			table_row = row.xpath('td//text()').extract()
			if len(table_row) == 0: # Skip empty row
				continue
			elif not table_row[1].isdigit(): # End of player stats
				break
			stats = Stats()
			stats['round_num'] = table_row[0]
			stats['draft_num'] = table_row[1]
			stats['draft_team'] = table_row[2]
			stats['name'] = table_row[3]
			stats['pos'] = table_row[4]
			stats['junior_team'] = table_row[5]
			stats['draft_year'] = re.findall('\d+',response.url)[0] # get year from url
			if len(table_row) == 12: # Player has played in the NHL
				stats['GP'] = table_row[6]
				stats['G'] = table_row[7]
				stats['A'] = table_row[8]
				stats['Pts'] = table_row[9]
				stats['PIM'] = table_row[10]
			else:
				stats['GP'] = 0
				stats['G'] = 0
				stats['A'] = 0
				stats['Pts'] = 0
				stats['PIM'] = 0
			yield stats