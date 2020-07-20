# NLP Summary2Dialogue

##   Character Pipeline
-    Character Detector :
    Used a combination of both traditional NER (NLTK) and neural net pretrained NER (spacy) to detect all possible characters incase any -one of them missed out some entity.  
-    Character Recogniser :
    To understand the characters and related context, find all mentions/expressions in the summary and link the pronouns with their nouns (Coreference Resolution).

##   Character Mapping
-    Here we extract dialogue from the script for each character and convert it into the corresponding vector using word2vec.
    You need to download this file to use the [word2vec model](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)

## Script GPT
-    We finetune GPT2 using data from imsdb. We use the imsdb_scrapper for scraping the data from the website.

##   example_1917
-    Example of using Finetuned GPT2 to generate scripts from summary
