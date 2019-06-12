#!/bin/bash

export PYTHON=python
export PYPY=python
if type "pypy" > /dev/null; then
	export PYPY=pypy
fi

RAW_TEXT='cuisine_data/Chinese.txt'
AUTO_LABEL=1
WORDNET_NOUN=0
# DATA_LABEL='cuisine_data/Chinese.label'
DATA_LABEL='cuisine_data/wiki.label.auto'
KNOWLEDGE_BASE='cuisine_data/wiki_labels_quality.txt'
KNOWLEDGE_BASE_LARGE='cuisine_data/wiki_labels_all.txt'

STOPWORD_LIST='cuisine_data/stopwords.txt'
SUPPORT_THRESHOLD=10

OMP_NUM_THREADS=4
DISCARD_RATIO=0.00
MAX_ITERATION=5

NEED_UNIGRAM=0
ALPHA=0.85

RESULTS='cuisine_results'

# clearance
rm -rf tmp
rm -rf cuisine_results

mkdir tmp
mkdir cuisine_results

# preprocessing
./bin/from_raw_to_binary_text ${RAW_TEXT} tmp/sentencesWithPunc.buf

# frequent phrase mining for phrase candidates
${PYPY} ./src/frequent_phrase_mining/main.py -thres ${SUPPORT_THRESHOLD} -o ./${RESULTS}/patterns.csv -raw ${RAW_TEXT}
${PYPY} ./src/preprocessing/compute_idf.py -raw ${RAW_TEXT} -o ${RESULTS}/wordIDF.txt

# feature extraction
./bin/feature_extraction tmp/sentencesWithPunc.buf ${RESULTS}/patterns.csv ${STOPWORD_LIST} ${RESULTS}/wordIDF.txt ${RESULTS}/feature_table_0.csv

if [ ${AUTO_LABEL} -eq 1 ];
then
	echo ===Auto Label Enable===
    ${PYTHON} src/classification/auto_label_generation.py ${KNOWLEDGE_BASE} ${KNOWLEDGE_BASE_LARGE} ${RESULTS}/feature_table_0.csv ${RESULTS}/patterns.csv ${DATA_LABEL}
else
	echo ===Auto Label Disable===
fi

# classifier training
./bin/predict_quality ${RESULTS}/feature_table_0.csv ${DATA_LABEL} ${RESULTS}/ranking.csv outsideSentence,log_occur_feature,constant,frequency 0 TRAIN ${RESULTS}/random_forest_0.model

MAX_ITERATION_1=$(expr $MAX_ITERATION + 1)

# 1-st round
./bin/from_raw_to_binary ${RAW_TEXT} tmp/sentences.buf
./bin/adjust_probability tmp/sentences.buf ${OMP_NUM_THREADS} ${RESULTS}/ranking.csv ${RESULTS}/patterns.csv ${DISCARD_RATIO} ${MAX_ITERATION} ./${RESULTS}/ ${DATA_LABEL} ./${RESULTS}/penalty.1

# 2-nd round
./bin/recompute_features ${RESULTS}/iter${MAX_ITERATION_1}_discard${DISCARD_RATIO}/length ${RESULTS}/feature_table_0.csv ${RESULTS}/patterns.csv tmp/sentencesWithPunc.buf ${RESULTS}/feature_table_1.csv ./${RESULTS}/penalty.1 1
./bin/predict_quality ${RESULTS}/feature_table_1.csv ${DATA_LABEL} ${RESULTS}/ranking_1.csv outsideSentence,log_occur_feature,constant,frequency 0 TRAIN ${RESULTS}/random_forest_1.model
./bin/adjust_probability tmp/sentences.buf ${OMP_NUM_THREADS} ${RESULTS}/ranking_1.csv ${RESULTS}/patterns.csv ${DISCARD_RATIO} ${MAX_ITERATION} ./${RESULTS}/1. ${DATA_LABEL} ./${RESULTS}/penalty.2

# phrase list & segmentation model
./bin/build_model ${RESULTS}/1.iter${MAX_ITERATION_1}_discard${DISCARD_RATIO}/ 6 ./${RESULTS}/penalty.2 ${RESULTS}/segmentation.model

if [ ${NEED_UNIGRAM} -eq 1 ];
then
	echo ===Unigram Enable===
	# unigrams
	normalize_text() {
	  awk '{print tolower($0);}' | sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
	  -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
	  -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
	  -e 's/«/ /g' | tr 0-9 " "
	}
	normalize_text < ${RESULTS}/1.iter${MAX_ITERATION}_discard${DISCARD_RATIO}/segmented.txt > tmp/normalized.txt

	cd word2vec_tool
	make
	cd ..
	./word2vec_tool/word2vec -train tmp/normalized.txt -output ./${RESULTS}/vectors.bin -cbow 2 -size 300 -window 6 -negative 25 -hs 0 -sample 1e-4 -threads ${OMP_NUM_THREADS} -binary 1 -iter 15
	time ./bin/generateNN ${RESULTS}/vectors.bin ${RESULTS}/1.iter${MAX_ITERATION_1}_discard${DISCARD_RATIO}/ 30 3 ${RESULTS}/u2p_nn.txt ${RESULTS}/w2w_nn.txt
	./bin/qualify_unigrams ${RESULTS}/vectors.bin ${RESULTS}/1.iter${MAX_ITERATION_1}_discard${DISCARD_RATIO}/ ${RESULTS}/u2p_nn.txt ${RESULTS}/w2w_nn.txt ${ALPHA} ${RESULTS}/unified.csv 100 ${STOPWORD_LIST}
else
	echo ===Unigram Disable===
	./bin/combine_phrases ${RESULTS}/1.iter${MAX_ITERATION_1}_discard${DISCARD_RATIO}/ ${RESULTS}/unified.csv
fi

${PYPY} src/postprocessing/filter_by_support.py ${RESULTS}/unified.csv ${RESULTS}/1.iter${MAX_ITERATION}_discard${DISCARD_RATIO}/segmented.txt ${SUPPORT_THRESHOLD} ${RESULTS}/salient.csv 
	
if [ ${WORDNET_NOUN} -eq 1 ];
then
    ${PYPY} src/postprocessing/clean_list_with_wordnet.py -input ${RESULTS}/salient.csv -output ${RESULTS}/salient.csv 
fi
