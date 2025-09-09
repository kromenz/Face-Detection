from icrawler.builtin import BingImageCrawler
import os

names = ["Arnold Schwarzenegger", "Tom Cruise", "Morgan Freeman", "Brad Pitt", "Dwayne Johnson", "Will Smith", "Tom Hanks", 
         "Denzel Washington", "Johnny Depp", "Robert Downey Jr.", "Mark Wahlberg", "Leonardo DiCaprio", "Bradley Cooper", 
         "Christian Bale", "Cristiano Ronaldo", "Lionel Messi", "Neymar Jr.", "Kylian Mbappe", "Obama", "Donald Trump", 
         "Bill Gates", "Beyonce", "Elon Musk", "Pavlidis"]

for name in names:
    out_dir = f"faces/{name.replace(' ', '_')}"
    os.makedirs(out_dir, exist_ok=True)

    crawler = BingImageCrawler(storage={'root_dir': out_dir})
    crawler.crawl(keyword=name, max_num=5)
