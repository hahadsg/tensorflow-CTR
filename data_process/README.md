# dataset

criteo dataset: [link](http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/)

* test 100k data

```shell
mkdir ../data/criteo/train100k
head -n100000 ../data/criteo/train.txt > ../data/criteo/train100k/train.txt
python criteo.py --input_dir=../data/criteo/train100k/ --output_dir ../data/criteo/train100k/
```

* all data

```shell
python criteo.py --input_dir=../data/criteo/ --output_dir ../data/criteo/
```
