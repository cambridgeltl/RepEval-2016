import getopt,sys,os
from nltk.tokenize import sent_tokenize

sys.path.insert(0, sys.path[0] + '/tools/')
from utilities import utilities

reload(sys)  
sys.setdefaultencoding('utf-8')



class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],'hi:n:qf')
        opts = dict(opts)

        if '-h' in opts:
            self.printHelp()

        if len(args) == 2:
            self.inputFile = args[0]
            self.outputfile = args[1]
        else:
            print >> sys.stderr, '\n*** ERROR: must specify precisely 2 arg input, output***'
            self.printHelp()
            
            
        if '-f' in opts:
            self.fname = True

    def printHelp(self):
        help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
        print >> sys.stderr, help
        exit()
        
if __name__ == "__main__":
    config = CommandLine()
    util = utilities()
    N = 100000
    for i, chunk in enumerate(util.read_in_chunks(config.inputFile, N)):
        SentenceList = []
        for item in chunk:
            try:
                item = item.encode('utf-8','ignore')
                result = sent_tokenize(item)
                #print "result", result
                SentenceList.append(result)
            except Exception:
                pass
           
        util.writeListOfList2File(config.outputfile, SentenceList)
    