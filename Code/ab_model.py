import os,random,math

"""
    The language model class implements a unigram, bigram and trigram model.
    The models are all initialised by running train().


 todo: update description for all models
"""
from nltk import word_tokenize as tokenize
import operator

class ab_model():

    """
        Initialises The model.
        trainingdir = the path of the training file
        files= the filenames of each training file used
        max_files = the amount of files used for training, this is for testing the system on small samples during development.
        limit_files = decides if the max files number is used.
    """
    def __init__(self,trainingdir, max_files = 5, limit_files = True):
        self.training_dir=trainingdir
        #limits the files used to max files if needed
        self.files= self.get_training(max_files,limit_files)
        print("Using {} files for training".format(len(self.files)))

    """
        used for testing to check that the amount of words has not changed when actions are performed such as creating uknown token
    """
    def sum_vals(self):
        bi_vals = 0
        tri_vals = 0

        for key,val in self.bigram.items():
            bi_vals += sum(val.values())

        for key,val in self.trigram.items():
            for k,v in val.items():
                tri_vals += sum(v.values())
        #print("unigram total values:{}".format(sum(self.unigram.values())))
        #print("bigram total values:{}".format(bi_vals))
        #print("trigram total values:{}".format(tri_vals))

    def merge_dicts(self,dict1,dict2):
        for k,v in dict2.items():
            dict1[k] = dict1.get(k,0)+ dict2.get(k,0)
        return dict1

    def get_training(self,max_files = 0, limit = False):
        #list of all training file names in file
        filenames=os.listdir(self.training_dir)
        #random.shuffle(filenames)
        if limit == False:
            return filenames
        else:
            #print(len(filenames[:max_files]))
            return filenames[:max_files]

    def train(self,unk_thresh = 2):
        self.unigram={}
        self.bigram={}
        self.trigram={}
        self._processfiles()
        #print("counts before uknowns for ab")
        #self.sum_vals()
        self.make_unknowns(known = unk_thresh)
        #print("counts after uknowns for ab")
        #self.sum_vals()
        self._discount()
        self._convert_to_probs()
        #print(self.trigram["__UNK"])

        """
            This method takes a list of tokens representing a sentence and proccesses it adding the counts to the unigram bigram
            and trigram models.
            todo:  change it all to proccess sentence.
        """
    def _processline(self,line):
        tokens=["__START"]+tokenize(line)+["__END"]
        #The bottom 2 lines for looking at the tokens before nth token immplement wraparound, so the token before start is end
        previous = tokens[len(tokens)-1]# token n -1
        pre_previous = tokens[len(tokens)-2]#token n - 2 used for trigram
        for token in tokens:

            #used to update the unigram dictionary
            self.unigram[token]=self.unigram.get(token,0)+1

            #next 3 lines are used to update the bigram dictionary
            current = self.bigram.get(previous,{})#gets the values for the word before the token being looked at
            current[token] = current.get(token,0) + 1 #increments the value for the current token in the previous tokens dictionary to +1
            self.bigram[previous] = current#sets the updated version

            #used to implement the trigram dictionary. same as bigram but with an extra level of depth
            minus_2 = self.trigram.get(pre_previous,{})
            minus_1 = minus_2.get(previous,{})
            minus_1[token] = minus_1.get(token,0) + 1
            minus_2[previous] = minus_1
            self.trigram[pre_previous] = minus_2

            pre_previous = previous#increments the n-2 token
            previous =token# increments the n-1


    """
        Proccesses each training file.
        todo: change to proccess by sentence rather than line.
    """
    def _processfiles(self):
        #iterates through each training file
        for afile in self.files:
            #print("Processing {}".format(afile))
            try:
                #opens file
                with open(os.path.join(self.training_dir,afile)) as instream:
                    #line number count is used to ignore the bloat in each file during training
                    line_num = 1
                    #iterares through each line and if that line is not a part of the bloat then it is proccessed.
                    for line in instream:
                        line=line.rstrip()
                        if len(line)>0and line_num > 250:
                            self._processline(line)
                        line_num +=1
            except UnicodeDecodeError:
                #print("UnicodeDecodeError processing {}: ignoring file".format(afile))
                print()

    """
    This method turns certain words into uknown tokens if they have been seen less times than the "known" variable.
    It works on the unigrams first then uses that data to act recursively through the bigram and trigram models.
    todo: test trigrams
    todo: comment trigram section
    """
    def make_unknowns(self,known=3):
        #runs through the unigram token count list and anything seen less than the known amount deleted and has its value added to the uknown token
        for (k,v) in list(self.unigram.items()):
            if v<known:
                del self.unigram[k]
                self.unigram["__UNK"]=self.unigram.get("__UNK",0)+v

        #recursively runs through bigram counts converting low frequency tokens to uknown tokens
        #the unigram known list is used to achieve this without having to recursively unpack the bigram counts into a total sum
        for (k,adict) in list(self.bigram.items()):#iterates through each token:dictionary pair
            for (kk,v) in list(adict.items()): #iterates through each token:value pair
                isknown=self.unigram.get(kk,0)
                if isknown< known: #token is unknown
                    adict["__UNK"]=adict.get("__UNK",0)+v #value added to unknown value
                    del adict[kk]
            isknown=self.unigram.get(k,0)
            if isknown< known:
                del self.bigram[k]
                self.bigram["__UNK"] = self.merge_dicts(self.bigram.get("__UNK",{}),adict)

            else:
                self.bigram[k]=adict
        #for trigrams
        for (k,dictdict) in list(self.trigram.items()):
            for (kk,adict) in list(dictdict.items()):
                for (kkk, v) in list(adict.items()):
                    isknown=self.unigram.get(kkk,0)
                    if isknown < known:
                        adict["__UNK"]=adict.get("__UNK",0)+v
                        del adict[kkk]
                isknown=self.unigram.get(kk,0)
                if isknown < known:
                    del self.trigram[k][kk]
                    self.trigram[k]["__UNK"] = self.merge_dicts(self.trigram.get(k,{}).get("__UNK",{}),adict)

                else:
                    self.trigram[k][kk]=adict

            isknown=self.unigram.get(k,0)
            if isknown< known:
                del self.trigram[k]
                current=self.trigram.get("__UNK",{})
                for k,v in dictdict.items():
                    current[k] = self.merge_dicts(current.get(k,{}),v)
                #current.update(dictdict)
                self.trigram["__UNK"]=current

            else:
                self.trigram[k]=dictdict
    """
        Converts the bottom values of the nested dictionary and the unigram counts into probabilities.
    """
    def _convert_to_probs(self):
        #converts unigram to a dict of token:P(token)
        self.unigram={k:v/sum(self.unigram.values()) for (k,v) in self.unigram.items()}
        #converts bigrams to a nested dict of (toke n-1:P(token n| token n-1))
        self.bigram={key:{k:v/sum(adict.values()) for (k,v) in adict.items()} for (key,adict) in self.bigram.items()}
        """
            converts trigrams into a 3 layer deep dict of { token n-2 : { token n-1: P(token n| token n-1) }}
            the line is quite long but i refrained from shortening it because it is essentially doing the same thing as the bigram probabilities
            except 1 extra nesting so still simple to understand.
        """
        self.trigram = {key:{key2:{k:v/sum(bidict.values()) for (k,v) in bidict.items()} for (key2,bidict) in tridict.items()} for (key,tridict) in self.trigram.items()}

    def _discount(self,discount=0.75):
        #discount each bigram count by a small fixed amount
        self.bigram={k:{kk:value-discount for (kk,value) in adict.items()}for (k,adict) in self.bigram.items()}

        for key,val in self.trigram.items():
            self.trigram[key]= {k:{kk:value-discount for (kk,value) in adict.items()}for (k,adict) in self.trigram[key].items()}

        #for each word, store the total amount of the discount so that the total is the same
        #i.e., so we are reserving this as probability mass
        for k in self.bigram.keys():
            lamb=len(self.bigram[k])
            self.bigram[k]["__DISCOUNT"]=lamb*discount

        for key in self.trigram.keys():
            for k in self.trigram[key].keys():
                lamb=len(self.trigram[key][k])
                self.trigram[key][k]["__DISCOUNT"]=lamb*discount

    """
        Retrieves the bottom layer probabilities. for unigram the probabilities is based on total words.
        For bigram and trigram the token probability is based on the context of the previously appearing words
    """
    def get_prob(self,token,context="",method="unigram"):
        if method=="unigram":
            return self.unigram.get(token,self.unigram.get("__UNK",0))
        elif method=="bigram":
            bigram=self.bigram.get(context[-1],self.bigram.get("__UNK",{}))
            big_p=bigram.get(token,bigram.get("__UNK",0))
            lmbda=bigram.get("__DISCOUNT",0)
            uni_p=self.unigram.get(token,self.unigram.get("__UNK",0))
            p=big_p+lmbda*uni_p
            return p
        elif method == "trigram":
            trigram = self.trigram.get(context[-2],self.trigram.get("__UNK",{}))
            bigram = trigram.get(context[-1],trigram.get("__UNK",{}))
            big_p=bigram.get(token,bigram.get("__UNK",0))
            lmbda=bigram.get("__DISCOUNT",0)
            uni_p=self.unigram.get(token,self.unigram.get("__UNK",0))
            p=big_p+lmbda*uni_p
            #print("context:{} token:{}".format(context,token))
            return p
        elif method == "trigram2":
            return (self.get_prob(token,context,method = "unigram")+self.get_prob(token,context,method = "bigram")+self.get_prob(token,context,method = "trigram"))
        else:
            #print("get_prob ERROR: token not found")
            return 0

    """
        todo: explain method
    """
    def compute_prob_line(self,line,method="unigram"):
        #this will add _start to the beginning of a line of text
        #compute the probability of the line according to the desired model
        #and returns probability together with number of tokens

        tokens=["__START"]+tokenize(line)+["__END"]
        acc=0
        for i,token in enumerate(tokens[1:]):
            acc+=math.log(self.get_prob(token,tokens[:i+1],method))
        return acc,len(tokens[1:])


    """
        todo:explain method
    """
    def compute_probability(self,filenames=[],method="unigram"):
        #computes the probability (and length) of a corpus contained in filenames
        if filenames==[]:
            filenames=self.files

        total_p=0
        total_N=0
        for i,afile in enumerate(filenames):
            #print("Processing file {}:{}".format(i,afile))
            try:
                with open(afile) as instream:
                    line_num = 1
                    for line in instream:
                        line=line.rstrip()
                        if len(line)>0and line_num > 250:
                            p,N=self.compute_prob_line(line,method=method)
                            total_p+=p
                            total_N+=N
                        line_num += 1
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing file {}: ignoring rest of file".format(afile))
        return total_p,total_N

    """
        todo:explain method
    """
    def compute_perplexity(self,filenames=[],method="bigram"):

        #compute the probability and length of the corpus
        #calculate perplexity
        #lower perplexity means that the model better explains the data

        p,N=self.compute_probability(filenames=filenames,method=method)
        #print(p,N)
        pp=math.exp(-p/N)
        return pp

    def data_stats(self):
        print("total words used: {}".format(sum(self.unigram.values())))
        print("unique words: {}".format(sum([1 for x in self.unigram.values()])))
        print("words used over 5 times: {}".format(sum([1 for x in self.unigram.values() if x > 4])))
