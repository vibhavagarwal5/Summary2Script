from scrapy.item import Item, Field

class ImsdbScraperItem(Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    source_url = Field()
    genre = Field()
    script_text = Field()
    title = Field()
