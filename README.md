# **Fast** top-**K** **A**rea **T**opics **E**xtraction (FastKATE)

This is the repository containing the source code, data and API used in our recent paper: Fast Top-*k* Area Topics Extraction (paper link will be available soon).

### Prerequisites
The following dependencies are required and must be installed separately:

- [Python3](https://www.anaconda.com/download/)
- [Aria2](https://aria2.github.io/) (used in our code for speeding up downloading Wikipedia dumps)

Then run `git clone https://github.com/thuzhf/FastKATE.git` to download this repository to your computer with the same name. From now on, we denote the **parent** directory of FastKATE on your computer as `<PARENT>`.

### Download and Preprocess Wikipedia Dumps

Since our model utilizes Wikipedia dumps, thus we need to download these data first.

1. Run `cd <PARENT>` to enter into the parent directory of FastKATE.

2. Run `python3 -m FastKATE.src.wiki_downloader <timestamp> <out_dir> <job>` to download wikipedia data from [wikidump](https://dumps.wikimedia.org/). For example, running `python3 -m FastKATE.src.wiki_downloader 20170901 ./wikidata/ all` will help you download all possibly needed data of Wikipedia up to 20170901 into the directory `./wikidata/`. You can only choose `timestamp`s that are listed on [Index of /enwiki/](https://dumps.wikimedia.org/enwiki/). For quick help, run: `python3 -m FastKATE.src.wiki_downloader -h`.

3. Decompress all downloaded files to the same directory with the same name (without suffixes).

4. We utilize [WikiExtractor](https://github.com/attardi/wikiextractor) to help extract plain text from Wikipedia dumps. Please download this tool first and then run: `python3 <WikiExtractor_dir>/WikiExtractor.py -o <wikidata_dir>/preprocessed/ -b 64M --no-templates <wikidata_dir>/enwiki-20170901-pages-articles-multistream.xml` to preprocess downloaded wikidata, where `<WikiExtractor_dir>` represents the directory of downloaded [WikiExtractor](https://github.com/attardi/wikiextractor), and `<wikidata_dir>` represents the directory of downloaded wikidata in step 2 (such as `/xx/xx/xx/wikidata/`).

### Generate Topic Embeddings

1. Run `cd <PARENT>` to enter into the parent directory of FastKATE.

2. Run `python3 -m FastKATE.src.topic_embeddings <timestamp> <wikidata_dir>` to extract candidate topics (in the form of phrases) from wikidump data and generate vector representations of each topic. For quick help, run: `python3 -m FastKATE.src.topic_embeddings -h`.

3. A pretrained topic embeddings model (which is trained using the wikidump of timestamp 20161201 and used in our paper) can be downloaded [here](https://mega.nz/#F!YNJTUCyb!TXy7Ju7c6kyPg5Q50zDzhQ) (including 3 files; you should download all 3 files and put them in the same folder if you want to use the pretrained model).

4. Actually our code can be easily modified to train topic embeddings on different datasets other than Wikipedia used here. For those who really want to do this, please refer to the source code for more details.

### Extract Category Structure from Wikipedia

- Run `cd <PARENT>` to enter into the parent directory of FastKATE.

- Run `python3 -m FastKATE.src.taxonomy <timestamp> <wikidata_dir>` to extract category structure from Wikipedia. For quick help, run: `python3 -m FastKATE.src.taxonomy -h`.

- A file containing extracted category structure can be downloaded [here](https://mega.nz/#F!kJITxQBL!XgsqoetqEazkm4W3tP_YXQ) (which is used in our paper).

### Fast top-K Area Topics Extraction (FastKATE) and its API

- Run `cd <PARENT>` to enter into the parent directory of FastKATE.

- Run `python3 -m FastKATE.src.api <wikidata_dir>` to run the extraction algorithm and set up the API. For quick help, run: `python3 -m FastKATE.src.api -h`. A currently running API can be visited [here](http://166.111.7.105:15400/topics?area=artificial_intelligence&k=15).
    - The inputs of the API are:
        - area: area name; should be lowercase; spaces should be replaced by `_`.
        - k: the number of topics needed to be extracted; should be a positive integer.

    - The output of the API is a dict in JSON format, which consists of:
        - area: the same as the input.
        - result: top-k extracted topics of the given area, accompanied and ranked (in descending order) by their relevance to the given area.
        - time: consumed time (in seconds).
