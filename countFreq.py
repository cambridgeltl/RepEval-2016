'''
Created on 22 Mar 2016

@author: Billy
'''
import getopt,sys,os,re
sys.path.insert(0, '../tools/')
from utilities import utilities
from collections import OrderedDict
import getopt,sys,os,re
util = utilities()

class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],'hi:n:qf')
        opts = dict(opts)

        if '-h' in opts:
            self.printHelp()

        if len(args) == 2:
            self.frequencyFile = args[0]
            self.simPairFile = args[1]
        else:
            print >> sys.stderr, '\n*** ERROR: must specify precisely 2 arg: Corpus Frequency Count , SimPairFile***'
            self.printHelp()
            
            
        if '-f' in opts:
            self.fname = True

    def printHelp(self):
        help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
        print >> sys.stderr, help
        exit()
        
def countFreq(fname, freqDict):
    
    N = 100000
    for i, chunk in enumerate(util.read_in_chunks(fname, N)):
        simDict = {}
        for item in chunk:
            #print item
            item = item.encode('utf-8')
            word1 = item.split("\t")[0]
            word2 = item.split("\t")[1]
            score = float(item.split("\t")[2])
            simDict[word1+":"+word2] = score
  
    sortsimDict = OrderedDict(sorted(simDict.items(), key=lambda t: t[1], reverse=True)) # 0 if sort by key
      
    print len(sortsimDict.keys())
    freqCountDict={}
    oovDict ={}
      
          
    for n, key in enumerate(sortsimDict.keys()):
        w1,w2 = key.split(":")
          
        if w1 in freqDict:
            w1Freq = freqDict[w1]
            freqCountDict[w1]=w1Freq
        else:
            w1Freq = "none"
            oovDict[w1] = 0
              
        if w2 in freqDict:
            w2Freq = freqDict[w2]
            freqCountDict[w2]=w2Freq
        else: 
            w2Freq = "none"
            oovDict[w2] = 0
        #print n+1,w1,w2, w1Freq, w2Freq
      
    group1 = 0 #[1, 100]
    group2 = 0 #[101, 1000] 
    group3 = 0 #[1001, 10000] 
    group4 = 0 #[10001 or above]
      
    for key, value in freqCountDict.items():
        if value <= 100 and value >=1:
            #print key, value, "1"
            group1+=1
        elif value >=101 and value <= 1000:
            #print key, value, "2"
            group2+=1
        elif value >=1001 and value <=10000:
            #print key, value, "3"
            group3+=1
        elif value >=10001:
            #print key, value, "4"
            group4+=1
              
    print "group1:", group1,"group2:", group2, "group3:", group3,"group4:", group4, "oov", len(oovDict.keys())
     
if __name__ == "__main__":
    config = CommandLine()
    
    freqDict = util.readFile2Dict(config.frequencyFile, " ")

    for root, dirs, files in os.walk(config.simPairFile):
        for file in files:
            if file.endswith(".txt"):
                print(os.path.join(root, file))
                
                try:
                    countFreq(os.path.join(root, file),freqDict)
                except Exception:
                    pass
            