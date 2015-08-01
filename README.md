# Clatern

[![Clojars Project](http://clojars.org/clatern/latest-version.svg)](http://clojars.org/clatern)

Machine learning in Clojure as easy as 1,2,3. This library is a work in progress and is currently not to be used for any serious purposes.

## Features

#### I/O:
- Read and write dataset from/to csv files

#### Preprocessing:
- Mean Normalizaion
- Min Max Scaling

#### Models:
- Linear Regression
- Logistic Regression
- Naive Bayes
- kNN
- KMeans Clustering

## Building and Running

#### Tests
Test can be run with:

    lein test

#### Jar
To build the Clatern library jar:

    lein jar

#### Uber Jar
To build the Clatern library uber jar:

    lein uberjar

#### Documentation
To generate the Clatern docs:

    lein doc

Documents will be located under the `doc` directory.

## License

Copyright © 2014 Rinu Boney

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
