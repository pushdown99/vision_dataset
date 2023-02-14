## download dataset from AI HUB (S3 bucket)

~~~console
$ cd download
$ chmod u+X download_dataset.sh
$ ./download_dataset.sh images 
$ ./download_dataset.sh bbox 
$ ./download_dataset.sh caption 
$ ./download_dataset.sh relation 
~~~

## build dataset

~~~console
$ python build.py prepare
$ python build.py build
~~~

## check files

~~~console
$ ls info
~~~
