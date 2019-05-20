# -*- coding: utf-8 -*-
from scrapy.spiders import CrawlSpider,Rule
from scrapy.linkextractors import LinkExtractor
from scout_spider.items import Player
import re

years = ['2018','2017','2016','2015','2014','2013','2012','2011']
current_year = '2019'

class PlayerSpider(CrawlSpider):
	name = 'player'
	allowed_domains = ['draftsite.com']
	start_urls = ['http://draftsite.com/nhl/draft-history/{}/'.format(year) for year in years]
	start_urls.append('https://draftsite.com/nhl/mock-draft/{}/'.format(current_year))
	rules = [Rule(LinkExtractor(allow='player'),callback='parse_report',follow=False),]

	def parse_report(self, response):
		player = Player()
		name = response.xpath('//*[@class="article-title"]//h1/text()').extract_first()
		report = response.xpath('//*[@class="fill_draft_info"]//td/text()').extract_first()
		draft_year = response.xpath('//*[@class="fill_draft_info"]//td/div/a/text()').extract_first()
		if draft_year is not None:
			draft_num = response.xpath('//*[@class="fill_draft_info"]//td/div/text()').extract()[2]
		else:
			draft_num = '0'
		if isinstance(report,str):
			report = report.rstrip()
			player['name'] = name
			player['report'] = report
			player['draft_year'] = draft_year.split()[0] if draft_year is not None else current_year
			player['draft_num'] = re.findall('\d+',draft_num)[0]
			return player