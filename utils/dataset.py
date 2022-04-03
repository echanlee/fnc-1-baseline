from csv import DictReader


class DataSet():
    def __init__(self, name="train", path="fnc-1"):
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        if name != "test" : 
            stances = name+"_stances.csv"

            self.stances = self.read(stances)
            #make the body ID an integer value
            for s in self.stances:
                s['Body ID'] = int(s['Body ID'])
            print("Total stances: " + str(len(self.stances)))
        articles = self.read(bodies)
        self.articles = dict()

        

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        
        print("Total bodies: " + str(len(self.articles)))



    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
