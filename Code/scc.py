import csv
from nltk import word_tokenize as tokenize
import random

class question:
    question = []
    missIndex = 0
    choices = []
    ans = ""
    ans_templates = ["a","b","c","d","e"]


    def __init__(self,aline):
        self.fields=aline
        self.question = tokenize(aline[1])
        self.choices = aline[2:]
        self.missIndex = self.question.index("_____")
        #print(self.question)
        #print(self.choices)
        #print(self.missI)
        if self.missIndex < 2: print("errrrrror")

    def get_ques(self,field):
        return self.question

    def add_answer(self,fields):
        self.answer=fields[1]


    def chooseUni(self,model):
        probs = [model.get_prob(choice) for choice in self.choices]
        return(self.ans_templates[probs.index(max(probs))])

    def chooseBi(self,model):
        cont = self.question[:self.missIndex]
        probs = [model.get_prob(choice,context = cont, method = "bigram") for choice in self.choices]
        return(self.ans_templates[probs.index(max(probs))])

    def chooseTri(self,model):
        cont = self.question[:self.missIndex]
        probs = [model.get_prob(choice,context = cont, method = "trigram") for choice in self.choices]
        return(self.ans_templates[probs.index(max(probs))])
    def chooseTri2(self,model):
        cont = self.question[:self.missIndex]
        probs = [model.get_prob(choice,context = cont, method = "trigram2") for choice in self.choices]
        return(self.ans_templates[probs.index(max(probs))])



    def predict(self,model,method):
        #eventually there will be lots of methods to choose from
        if method=="chooseRand":
            return (random.choice(["a","b","c","d","e"]))
        elif method=="unigram":
            return self.chooseUni(model)
        elif method=="bigram":
            return self.chooseBi(model)
        elif method=="trigram":
            return self.chooseTri(model)
        elif method=="trigram2":
            return self.chooseTri2(model)
        else:
            print("Error. Uknown method.")

    def predict_and_score(self,model,method):
        #compare prediction according to method with the correct answer
        #return 1 or 0 accordingly
        prediction=self.predict(method=method,model = model)
        if prediction ==self.answer:
            return 1
        else:
            return 0

class scc_reader:

    def __init__(self,qs,ans):
        self.qs=qs
        self.ans=ans
        self.read_files()

    def read_files(self):

        #read in the question file
        with open(self.qs) as instream:
            csvreader=csv.reader(instream)
            qlines=list(csvreader)

        #store the column names as a reverse index so they can be used to reference parts of the question
        question.colnames={item:i for i,item in enumerate(qlines[0])}

        #create a question instance for each line of the file (other than heading line)
        self.questions=[question(qline) for qline in qlines[1:]]

        #read in the answer file
        with open(self.ans) as instream:
            csvreader=csv.reader(instream)
            alines=list(csvreader)

        #add answers to questions so predictions can be checked
        for q,aline in zip(self.questions,alines[1:]):
            q.add_answer(aline)

    def get_field(self,field):
        return [q.get_field(field) for q in self.questions]

    def predict(self,method="chooseRand"):
        return [q.predict(method=method,model = model) for q in self.questions]

    def predict_and_score(self,method="chooseRand", model = None):
        if model == None: print("Error. No model given.")
        scores=[q.predict_and_score(method=method,model = model) for q in self.questions]
        #print("Method: {} Score: {}".format(method,(sum(scores)/len(scores))))
        return sum(scores)/len(scores)
