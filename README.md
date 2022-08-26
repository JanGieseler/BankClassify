# BankClassify - automatically classify your bank statement entries

Originaly forked from https://github.com/robintw/BankClassify and the modified to match my needs


## run with docker

`docker run -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/scipy-notebook`



### structure

raw exported data is put into the `raw` folder. The the function `add_raw_data` reads in new data, checks for duplicates with existing data and appends new data to the clean data, where each year is kept in a individual file for each source.
The classifier is then trained on the clean data.

```
data
├── raw
│   └── paypal
        └── raw_data_1.csv
        └── raw_data_2.csv
        └── ...
│   └── another_source
        └── raw_data_1.csv
        └── raw_data_2.csv
        └── ...
├── clean
│   ├── 2020_paypal.csv
│   └── ..
```