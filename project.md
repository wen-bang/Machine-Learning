Background
----------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, the goal is
to use data from accelerometers on the belt, forearm, arm, and dumbell
of 6 participants. They were asked to perform barbell lifts correctly
and incorrectly in 5 different ways. More information is available from
[the website
here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).

Goal
----

The goal of this project is to predict the manner in which they did the
exercise. This is the "classe" variable in the training set.

Data
----

The training data for this project are available
[here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv).

The test data are available
[here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

The data for this project come from this
[source](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).

Reference
---------

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H.
Wearable Computing: Accelerometers' Data Classification of Body Postures
and Movements. Proceedings of 21st Brazilian Symposium on Artificial
Intelligence. Advances in Artificial Intelligence - SBIA 2012. In:
Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer
Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI:
10.1007/978-3-642-34459-6\_6. Cited by 2 (Google Scholar)

My Work
-------

### Load the library, data and do some preprocessing

    library(caret)

    ## Warning: package 'caret' was built under R version 3.4.4

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.4.4

    training = read.csv("pml-training.csv")
    str(training)

From the result of *str(training)*, we found that the missing values
are: **""**, **NA** and **"\#DIV/0!"**. So let's reload the data.

    training = read.csv("pml-training.csv",na.strings = c("","NA","#DIV/0"))
    testing = read.csv("pml-testing.csv",na.strings = c("","NA","#DIV/0"))
    dim(training)

    ## [1] 19622   160

Also, we noticed that the first 7 columns are not related to the
response *classe*. So we exclude them from both training and testing
data sets.

    training = training[,8:ncol(training)]
    testing = testing[,8:ncol(testing)]

### Cross Validation

There are enough observations in the training data set to separate them
into a subtraining data and a subtesting data.

    set.seed(11231)
    intrain = createDataPartition(training$classe,p=0.5,list=F)
    subtraining = training[intrain,]
    subtesting = training[-intrain,]

### Dealing with missing value columns

    NAs = apply(subtraining,2,function(x){sum(is.na(x))})
    table(NAs)

    ## NAs
    ##    0 9614 
    ##   53  100

For all the columns, either it does not contain a missing value, or it
contains **9614** missing values! Note that each column contains only
**9812** observations, which is to say, columns with missing values miss
the most of data -- so we throw those columns away rather than imputing.

    subtesting = subtesting[,colSums(is.na(subtraining))==0]
    testing = testing[,colSums(is.na(subtraining))==0]
    subtraining = subtraining[,colSums(is.na(subtraining))==0]

Keep in mind that all data processing should be done in the same way to
both training and testing.

### Model1: **boosting** model using **gbm**

    library(gbm)

    ## Warning: package 'gbm' was built under R version 3.4.4

    ## Loading required package: survival

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.3

    model = gbm(classe~.,data=subtraining,distribution='multinomial',n.trees=4000)
    pred = predict(model,newdata=subtraining,n.trees=4000,type='response')
    predict = as.factor(colnames(pred)[apply(pred,1,which.max)])
    confusionMatrix(predict,subtraining$classe)$overall[1]

    ##  Accuracy 
    ## 0.7328781

Only more than 70% accuracy -- and this is the training accuracy! Let's
check the subtesting set accuracy.

    pred = predict(model,newdata=subtesting,n.trees=4000,type='response')
    predict = as.factor(colnames(pred)[apply(pred,1,which.max)])
    confusionMatrix(predict,subtesting$classe)$overall[1]

    ##  Accuracy 
    ## 0.7325178

### Model2: **random forest** model using **randomForest**

    library(randomForest)

    ## Warning: package 'randomForest' was built under R version 3.4.4

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    model_rf = randomForest(classe~.,data=subtraining,type="class")
    confusionMatrix(predict(model_rf,subtraining),subtraining$classe)$overall[1]

    ## Accuracy 
    ##        1

The training accuracy is full. Let's check the subtesting set accuracy.

    confusionMatrix(predict(model_rf,subtesting),subtesting$classe)$overall[1]

    ##  Accuracy 
    ## 0.9904179

Still high -- so the out of sample error is very small -- so we choose
this random forest model to predict on the testing set.

    prediction = predict(model_rf,testing)

Conclusion
----------

The random forest model seems perfect for this data and our prediction
on the test data are:

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
