# **Fast** top-**K** **A**rea **T**opics **E**xtraction (FastKATE)

This repository contains the source code, data and API used in our recent paper: Fast Top-*k* Area Topics Extraction (paper link will be available soon).

### Prerequisites
The following dependencies are required and must be installed separately:

- [Python 3](https://www.anaconda.com/download/) (used to run our programs)
- [Aria2](https://aria2.github.io/) (used to speed up downloading Wikipedia dumps)
- [WikiExtractor](https://github.com/attardi/wikiextractor) (used to extract plain text from Wikipedia dumps)

Then run `git clone https://github.com/thuzhf/FastKATE.git` to download this repository to your computer (with the same name). For convenience, please put WikiExtractor and FastKATE under the same parent directory, and we denote this parent directory as `<PARENT>` in the following steps.

### Download and Preprocess Wikipedia Dumps

Since our model utilizes Wikipedia dumps, thus we need to download these data first. We choose Wikipedia dumps of timestamp `20170901` as our example in the following steps. Available timestamps can be found [here](https://dumps.wikimedia.org/enwiki/).

1. Run `cd <PARENT>` to enter into the parent directory of FastKATE.

2. Run `python3 -m FastKATE.src.wiki_downloader 20170901 <PARENT>/wikidata/ all` will help you download all possibly needed data of Wikipedia of timestamp `20170901` into the directory `<PARENT>/wikidata/`. For quick help, run `python3 -m FastKATE.src.wiki_downloader -h`.

3. Decompress all downloaded Wikipedia dumps to the `<PARENT>/wikidata/` with the same name (without suffixes such as `.gz` and `.bz2`).

4. Run `python3 <PARENT>/wikiextractor/WikiExtractor.py -o <PARENT>/wikidata/preprocessed/ -b 64M --no-templates <PARENT>/wikidata/enwiki-20170901-pages-articles-multistream.xml` to preprocess downloaded wikidata. For quick help, run `python3 <PARENT>/wikiextractor/WikiExtractor.py -h`.

### Generate Topic Embeddings

1. Run `cd <PARENT>` to enter into the parent directory of FastKATE.

2. Run `python3 -m FastKATE.src.topic_embeddings 20170901 <PARENT>/wikidata/` to extract candidate topics (in the form of phrases) from wikidump data and generate vector representations of each topic. For quick help, run: `python3 -m FastKATE.src.topic_embeddings -h`.

3. A pretrained topic embeddings model (which is trained using the wikidump of timestamp 20161201 and used in our paper) can be downloaded [here](https://mega.nz/#F!YNJTUCyb!TXy7Ju7c6kyPg5Q50zDzhQ) (including 3 files; you should download all 3 files and put them in the same folder if you want to use the pretrained model).

4. Actually our code can be easily modified to train topic embeddings on different datasets other than Wikipedia used here. For those who really want to do this, please refer to the source code for more details.

### Extract Category Structure from Wikipedia

1. Run `cd <PARENT>` to enter into the parent directory of FastKATE.

2. Run `python3 -m FastKATE.src.taxonomy 20170901 <PARENT>/wikidata/` to extract category structure from Wikipedia. For quick help, run: `python3 -m FastKATE.src.taxonomy -h`.

3. A file containing extracted category structure can be downloaded [here](https://mega.nz/#F!kJITxQBL!XgsqoetqEazkm4W3tP_YXQ) (which is used in our paper).

### Fast top-K Area Topics Extraction (FastKATE) and its API

1. Run `cd <PARENT>` to enter into the parent directory of FastKATE.

2. Run `python3 -m FastKATE.src.api <PARENT>/wikidata/` to run the extraction algorithm and set up the API. For quick help, run: `python3 -m FastKATE.src.api -h`. A currently running API can be visited [here](http://166.111.7.105:15400/topics?area=artificial_intelligence&k=15).
    - The inputs of the API are:
        - area: area name; should be lowercase; spaces should be replaced by `_`.
        - k: the number of topics needed to be extracted; should be a positive integer.

    - The output of the API is a dict in JSON format, which consists of:
        - area: the same as the input.
        - result: top-k extracted topics of the given area, accompanied and ranked (in descending order) by their relevance to the given area.
        - time: consumed time (in seconds).
