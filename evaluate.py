# -*- coding: utf-8 -*-
'''
Created on 4 Dec 2015

@author: Billy
'''

import os

from wvlib import wvlib
from wvlib import evalrank as eva
import sys
from mailbox import FormatError
reload(sys)  
import getopt

sys.setdefaultencoding('utf8')


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
            

    def printHelp(self):
        help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
        print >> sys.stderr, help
        exit()     
                
    
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
    
    from tools import utilities as util
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
                     
    