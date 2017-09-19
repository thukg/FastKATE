# **Fast** top-**K** **A**rea **T**opics **E**xtraction (FastKATE)

This is the repository containing the source code, data and API used in our recent paper: Fast Top-*k* Area Topics Extraction (paper link will be available soon).

### Prerequisites
The following dependencies are required and must be installed separately:

- [Python3](https://www.anaconda.com/download/)
- [Aria2](https://aria2.github.io/) (used in our code for speeding up downloading Wikipedia dumps)

Then run `git clone https://github.com/thuzhf/FastKATE.git` to download this repository to your computer with the same name. From now on, we denote the **parent** directory of FastKATE on your computer as `<PARENT>`.

### Download and Preprocess Wikipedia Dumps

- Run `cd <PARENT>` to enter into the parent directory of FastKATE.

- Run `python3 -m FastKATE.src.wiki_downloader [time_stamp] [out_dir] [job]` to download wikipedia data from [wikidump](https://dumps.wikimedia.org/). For example, running `python3 -m FastKATE.src.wiki_downloader 20170901 ./wikidata/ all` will help you download all possibly needed data of Wikipedia up to 20170901 into the directory `./wikidata/`. You can only choose `time_stamp`s that are listed on [Index of /enwiki/](https://dumps.wikimedia.org/enwiki/). Please refer to the following for more details on parameters:
~~~~
usage: wiki_downloader.py [-h] time_stamp out_dir job

positional arguments:
  time_stamp  such as 20170901
  out_dir     output directory
  job         all_titles/pages_articles/pages_articles_multistream/pages/categ
              orylinks/category/all

optional arguments:
  -h, --help  show this help message and exit
~~~~

- Decompress all downloaded files to the same directory with the same name (without suffixes).

- We utilize [WikiExtractor](https://github.com/attardi/wikiextractor) to help extract plain text from Wikipedia dumps. Please download this tool and run a command like this: `python3 <WikiExtractor_dir>/WikiExtractor.py -o <wikidata_dir>/wikidump/ -b 64M --no-templates <wikidata_dir>/enwiki-20170901-pages-articles-multistream.xml` to preprocess downloaded wikidata, where `<WikiExtractor_dir>` represents the directory of downloaded [WikiExtractor](https://github.com/attardi/wikiextractor), and `<wikidata_dir>` represents the directory of downloaded wikidata in the previous step. After this step, all preprocessed wikidata will be in `<wikidata_dir>/wikidump/`.

### Generate Topic Embeddings

- Run `cd <PARENT>` to enter into the parent directory of FastKATE.

- Open file `FastKATE/src/topic_embeddings.py`, and change the value of the variable `timestamp` to your actual value.

- Run `python3 -m FastKATE.src.topic_embeddings` to extract candidate topics (in the form of phrases) from wikidump data and generate vector representations of each topic. (If you haven't changed any default parameter settings up to now, you don't need to change the default parameter settings here either.)
~~~~
usage: topic_embeddings.py [-h] [--document_dir DOCUMENT_DIR]
                           [--document_cleaned_dir DOCUMENT_CLEANED_DIR]
                           [--category_file CATEGORY_FILE]
                           [--category_titles_file CATEGORY_TITLES_FILE]
                           [--page_titles_file PAGE_TITLES_FILE]
                           [--raw_topics RAW_TOPICS]
                           [--alphanumeric_topics ALPHANUMERIC_TOPICS]
                           [--new_sentences_dir NEW_SENTENCES_DIR]
                           [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --document_dir DOCUMENT_DIR
                        the directory of preprocessed (by WikiExtractor)
                        wikidump data (default: ./wikidata/wikidump/)
  --document_cleaned_dir DOCUMENT_CLEANED_DIR
  --category_file CATEGORY_FILE
                        wiki category file downloaded using wiki_downloader.py
                        (default: ./wikidata/enwiki-20170901-category.sql)
  --category_titles_file CATEGORY_TITLES_FILE
  --page_titles_file PAGE_TITLES_FILE
  --raw_topics RAW_TOPICS
  --alphanumeric_topics ALPHANUMERIC_TOPICS
                        this output file will contain all candidate topics (in
                        the form of phrases) (default:
                        ./wikidata/enwiki-20170901-pages-categories-titles-
                        alphanumeric)
  --new_sentences_dir NEW_SENTENCES_DIR
  --model_path MODEL_PATH
~~~~

- A pretrained topic embeddings model (which is trained using the wikidump of timestamp 20161201 and used in our paper) can be downloaded [here](https://mega.nz/#F!YNJTUCyb!TXy7Ju7c6kyPg5Q50zDzhQ) (including 3 files; you should download all 3 files and put them in the same folder if you want to use the pretrained model).

- In fact, you can use the our code to train topic embeddings on different datasets other than wikipedia used here. If you want to do this, you only need to ensure that:
    - `document_dir` can contain arbitrary deep directories; each line of any descendant file under this directory should be a paragraph of normal text (can contain punctuations).
    - You can provide the file of `raw_topics` directly and comment the code of step 2~3 in the `main()` function of `topic_embeddings.py`. Each line of the file of `raw_topics` should be a candidate topic (phrase) where spaces should be replaced by `_`.

### Extract Category Structure from Wikipedia

- Run `cd <PARENT>` to enter into the parent directory of FastKATE.

- Run `python3 -m FastKATE.src.taxonomy` to extract category structure from Wikipedia, which will be stored in the file `./wikidata/taxonomy_lemmatized.pkl` (if you do not change any default parameter settings).
~~~~
usage: taxonomy.py [-h] [--pages_infile PAGES_INFILE]
                   [--pages_outfile PAGES_OUTFILE]
                   [--categorylinks_infile CATEGORYLINKS_INFILE]
                   [--categorylinks_outfile CATEGORYLINKS_OUTFILE]
                   [--category_infile CATEGORY_INFILE]
                   [--category_outfile CATEGORY_OUTFILE]
                   [--page_and_category_outfile PAGE_AND_CATEGORY_OUTFILE]
                   [--taxonomy_outfile TAXONOMY_OUTFILE]
                   [--taxonomy_lemmatized_outfile TAXONOMY_LEMMATIZED_OUTFILE]

optional arguments:
  -h, --help            show this help message and exit
  --pages_infile PAGES_INFILE
  --pages_outfile PAGES_OUTFILE
  --categorylinks_infile CATEGORYLINKS_INFILE
  --categorylinks_outfile CATEGORYLINKS_OUTFILE
  --category_infile CATEGORY_INFILE
  --category_outfile CATEGORY_OUTFILE
  --page_and_category_outfile PAGE_AND_CATEGORY_OUTFILE
  --taxonomy_outfile TAXONOMY_OUTFILE
  --taxonomy_lemmatized_outfile TAXONOMY_LEMMATIZED_OUTFILE
~~~~

- A file containing extracted category structure can be downloaded [here](https://mega.nz/#F!kJITxQBL!XgsqoetqEazkm4W3tP_YXQ) (which is used in our paper).

### Fast top-K Area Topics Extraction (FastKATE) and its API

- Run `cd <PARENT>` to enter into the parent directory of FastKATE.

- Run `python3 -m FastKATE.src.api` to run the extraction algorithm and set up the API. A currently running API can be visited [here](http://166.111.7.105:15400/topics?area=artificial_intelligence&k=15).
    - The inputs of the API are:
        - area: area name; should be lowercase; spaces should be replaced by `_`.
        - k: the number of topics needed to be extracted; should be a positive integer.

    - The output of the API is a dict in JSON format, which consists of:
        - area: the same as the input.
        - result: top-k extracted topics of the given area, accompanied and ranked (in descending order) by their relevance to the given area.
        - time: consumed time (in seconds).
