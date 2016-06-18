###############################################################################################
# Modified of word2vec demo-train-big-data.sh
# 
# 
#
# Downloads about 8 billion words, create raw Corpus
# 
#
###############################################################################################


mkdir Bigdata
cd Bigdata

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz
wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz

gzip -d news.2012.en.shuffled.gz 
gzip -d news.2013.en.shuffled.gz  
tar -xvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz 
tar -zxvf umbc_webbase_corpus.tar.gz webbase_all/*.txt 
 

cat < news.2012.en.shuffled > data.txt 
cat < news.2013.en.shuffled > data1.txt  


for i in `ls 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled`; do
  cat < 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/$i >> data2.txt
done
 
for i in `ls webbase_all`; do
  cat < webbase_all/$i >> data3.txt
done

wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
python ../wikiextractor/WikiExtractor.py -cb 250K -o extracted $1
find extracted -name '*bz2' -exec bzip2 -dc {} \; > data4.txt
#rm -rf extracted 

cat data.txt data1.txt data2.txt data3.txt data4.txt > big-data_corpus.txt