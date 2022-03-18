import scrapy


class AdevarulSpider(scrapy.Spider):
    name = 'adevarul'
    allowed_domains = ['adevarul.ro']

    def start_requests(self):
        start_urls = ['http://adevarul.ro/']
        for url in start_urls:
            yield scrapy.Request(url = url, callback = self.parse)


    def parse(self, response):
        for link in response.css('h3.title a,h3.defaultTitle a'):
            yield response.follow(link, self.parse_article)

    def parse_article(self, response):
        items = response.css('p[data-index]')
        textItems = [item.get() for item in items]
        if (len(textItems) > 0):
            text = ' '.join(textItems)
            yield {
                'text': text
            }
