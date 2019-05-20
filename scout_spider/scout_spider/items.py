# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class Player(scrapy.Item):
	name = scrapy.Field()
	report = scrapy.Field()
	draft_year = scrapy.Field()
	draft_num = scrapy.Field()

class Stats(scrapy.Item):
	name = scrapy.Field()
	round_num = scrapy.Field()
	draft_num = scrapy.Field()
	draft_team = scrapy.Field()
	pos = scrapy.Field()
	junior_team = scrapy.Field()
	GP = scrapy.Field()
	G = scrapy.Field()
	A = scrapy.Field()
	Pts = scrapy.Field()
	PIM = scrapy.Field()
	draft_year = scrapy.Field()