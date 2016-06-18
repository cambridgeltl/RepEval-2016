# -*- coding: utf-8 -*-
'''
Created on 4 Dec 2015

@author: Billy
'''
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
# from difflib import SequenceMatcher
#from mentionsLocator import *
#import itertools
import os
#from gensim.models import *
from wvlib import wvlib
# import re
# from wvlib.evalclass import score
# import numpy as np  # Make sure that numpy is imported
# from operator import itemgetter
# from oboParser import *
# from wvlib import similarity
from wvlib import evalrank as eva
# from collections import defaultdict
# import csv
import sys
from mailbox import FormatError
reload(sys)  
# import numpy
# import gzip
# import math
# import sys, re
import getopt

sys.setdefaultencoding('utf8')

# def read_word_vecs(filename):
#   wordVectors = {}
#   if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
#   else: fileObject = open(filename, 'r')
#   
#   for line in fileObject:
#     line = line.strip().lower()
#     word = line.split()[0]
#     wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
#     for index, vecVal in enumerate(line.split()[1:]):
#       wordVectors[word][index] = float(vecVal)
#     ''' normalize weight vector '''
#     wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
#     
#   sys.stderr.write("Vectors read from: "+filename+" \n")
#   return wordVectors
  
# def convertOboToDict(oboDict):
#     oboDict=sorted(oboDict.items(), key=itemgetter(0))
#     dict={}
#     for key,value in oboDict:
#         if key in dict:
#             dict[key].append(value['synonym'])
#         else:
#             dict[key] = value['synonym']
#     return dict
# 
# def convertEllendorff2015ToDict(lexiconList,choice):
#     f=csv.DictReader(open(lexiconList),delimiter='\t')
#     termDict=None
# 
#     if choice.lower() == "dict":
#         termDict = defaultdict(set)
#         
#         for row in f:
#             term=row['term'].strip()
#             preferred_term=row['preferred_term'].strip()                
#             #print term,preferred_term
#             if not term==preferred_term and not term=="":
#                 termDict[term].add(preferred_term)
#                 #print termDict[term],row['preferred_term'].strip()
#     else: 
#         if choice.lower() == "list":
#             termDict=[]
#             
#             for row in f:
#                 term=row['term'].strip()
#                 preferred_term=row['preferred_term'].strip()
#                 if not term==preferred_term and not term=="":
#                     termDict.append((term,preferred_term))
#             
#     return termDict
# 
# def combineSynonym(testList,path):
#     res=[]
#     mergeProcess=1
#     while not mergeProcess==0:
#         mergeProcess=0
#         for checkList in testList:
#             for item1 in testList:
#                 if len(list(set(checkList) & set(item1))):
#                     newList= list(set(checkList) | set(item1))
#                     res.append(newList)
#                     mergeProcess+=1
                
class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],'hi:n:qf')
        opts = dict(opts)

        if '-h' in opts:
            self.printHelp()

        if '-i' in opts:
            self.inputFile = opts['-i']
            #self.outputFile = args[1]
        else:
            print >> sys.stderr, '\n*** ERROR: must specify precisely 1 arg with -i***'
            self.printHelp()
            
#         if '-o' in opts:
#             self.interp_points = int(opts['-i'])
#         else:
#             self.interp_points = 10
# 
#         if '-n' in opts:
#             self.response_limit = int(opts['-n'])
#         else:
#             self.response_limit = None


    def printHelp(self):
        help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
        print >> sys.stderr, help
        exit()     
                
# def evaluateFile(fname):
#     wv = wvlib.load(fname).normalize()
#     evafilePath=[os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/MayoSRS', 'MayoSRS.txt'),os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/UMNSRS', 'UMNSRS-sim.txt')]#,os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/UMNSRS', 'UMNSRS-sim.txt')
#     print "Finish load w2v model"
#     references = [(r, eva.read_referenceSingleWords(r)) for r in evafilePath]
#     print '%20s\trho\tmissed\ttotal\tratio' % 'dataset'
#     for name, ref in references:
#         #rho, count = eva.evaluateTest(newWordVecs, ref,wordList)
#         rho, count = eva.evaluate(wv, ref)
#         total, miss = len(ref), len(ref) - count
#         
#         print '%20s\t%.4f\t%d\t%d\t(%.2f%%)' % \
#             (eva.baseroot(name), rho, miss, total, 100.*miss/total)
    
if  __name__ =='__main__':

    
    #filePath=os.path.join(os.path.dirname(__file__), 'w2vData', 'PubMed15_Dependancy1.txt') #PubMed-w2v.bin #PubMed-and-PMC-w2v.bin
    
    evafilePath=[os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/SimLex-999', 'SimLex-999.txt'),\
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/WordSim-353', 'WordSim-353.txt'),\
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/MC-30', 'EN-MC-30.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/MEM-TR-3k', 'EN-MEN-TR-3k.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/MTurk-287', 'EN-MTurk-287.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/MTurk-771', 'EN-MTurk-771.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/RG-65', 'EN-RG-65.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/RW-STANFORD', 'EN-RW-STANFORD.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/VERB-143', 'EN-VERB-143.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/WS-353-REL', 'EN-WS-353-REL.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/WS-353-SIM', 'EN-WS-353-SIM.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/WS-353-ALL', 'EN-WS-353-ALL.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/YP-130', 'EN-YP-130.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/MayoSRS', 'MayoSRS.txt'),
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/UMNSRS', 'UMNSRS-sim.txt'),\
                 os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/UMNSRS', 'UMNSRS-rel.txt')]
    
#     evafilePath=[os.path.join(os.path.dirname(__file__), 'word2Vec/data/', 'sorted-by-min-split-aa'),os.path.join(os.path.dirname(__file__), \
#                                                                                                                   'word2Vec/data/', 'sorted-by-min-split-ab')]#,os.path.join(os.path.dirname(__file__), 'wvlib/word-similarities/UMNSRS', 'UMNSRS-sim.txt')
    
    from tools import utilities as util
    #lexiconFile=os.path.join(os.path.dirname(__file__), 'lexicon/', 'Ellendorff2015.csv')
    #outputFile=os.path.join(os.path.dirname(__file__), 'lexicon/', 'lexiconList_Normal.pkl')
    #vectorTxtfile=os.path.join(os.path.dirname(__file__), 'sparse-coding/vector/', 'output_vecs_10000_outsize_10.txt')
    #testFile=os.path.join(os.path.dirname(__file__), 'retrofit/lexicons/', 'Lex_lower_UMNSRS_trim.txt')
    #oboDict=oboParser('pro.obo').terms
    #print "Finish load obo file"
    #replace word2Vec with wvlib_Billy1
    config = CommandLine()
    #from word2Vec import tools as util
    if os.path.isfile(config.inputFile):
        try:
            wv = wvlib.load(config.inputFile).normalize()
            #references = [(r, eva.read_referenceSingleWords(r)) for r in evafilePath]
            references = [(r, eva.read_reference(r)) for r in evafilePath]
            print '%20s\trho\tmissed\ttotal\tratio' % 'dataset'
            for name, ref in references:
                #rho, count = eva.evaluateTest(newWordVecs, ref,wordList)
                rho, count = eva.evaluate(wv, ref)
                total, miss = len(ref), len(ref) - count
                print '%20s\t%.4f\t%d\t%d\t(%.2f%%)' % \
                (eva.baseroot(name), rho, miss, total, 100.*miss/total)
        except FormatError:
            print "skip",config.inputFile
    else:
            folderList=util.get_filepaths(config.inputFile)
            for i,item in enumerate(folderList):
                filename, file_extension = os.path.splitext(item)
                #print i,item
                if  ".DS_Store" not in item:
                    try:
                        wv = wvlib.load(item).normalize()
                        references = [(r, eva.read_referenceSingleWords(r)) for r in evafilePath]
                        print '%20s\trho\tmissed\ttotal\tratio' % 'dataset'
                        for name, ref in references:
                            #rho, count = eva.evaluateTest(newWordVecs, ref,wordList)
                            rho, count = eva.evaluate(wv, ref)
                            total, miss = len(ref), len(ref) - count
                            print '%20s\t%.4f\t%d\t%d\t(%.2f%%)' % \
                            (eva.baseroot(name), rho, miss, total, 100.*miss/total)
                    except FormatError:
                        print "skip",item
                     
    #model = Word2Vec.load_word2vec_format(filePath, binary=True)   
    #print "Finish load w2v model" ,filePath

    #lexiconList=convertOboToDict(oboDict)
    #lexiconList=convertEllendorff2015ToDict(lexiconFile,"dict")
    #
    
    #lexiconList= rf.retrofit.read_lexiconList(testFile)
    #print "lexicon1 size" ,len(lexiconList1)
    #lexiconList=test.readFile(outputFile)
    
    
    
    #print "finish create lexicon list"
    #newWordVecs,wordList=rf.retrofit.retrofit(wv.word_to_vector_mapping(),lexiconList , 5)
    #print "finish create new word model","\n"
    
    #newWordVecs=rf.retrofit.read_word_vecs(vectorTxtfile)
    ##wordList=rf.retrofit.read_lexicon(testFile)



# original eva code 
#     print '%20s\trho\tmissed\ttotal\tratio' % 'dataset'
#     for name, ref in references:
#         rho, count = eva.evaluate(newWordVecs, ref)
#         total, miss = len(ref), len(ref) - count
#         print '%20s\t%.4f\t%d\t%d\t(%.2f%%)' % \
#             (eva.baseroot(name), rho, miss, total, 100.*miss/total)

 
    