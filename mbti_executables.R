#--------------- Analyzing the MBTI Personality Data - Executables ---------------
#
# Goal: predict personality labels based on raw data.
# Data: 8675 rows of personality profiles from the MBTI and their last posts
# on some forum.
# Features: research linked to social media behavior based on personality &
# linguistic features
# Method: SVM using n-grams/tf.idf etc.

# Lisa Gotzian, March 2019

#---------------------- Preliminaries ----------------------------
source("mbti_functions.R")
library(tm)
library(plyr)
library(magrittr)
library(quanteda.dictionaries) # for linguistic features
library(quanteda)
library(ngram)
library(RTextTools) # for the svm process
library(ggplot2) # for a heatmap
library(reshape2) # to melt data
library(caret) # for a confusion matrix

# As it has errors, I adjusted the original library function from RTextTools
# and took out lines 42 and 43 using:
trace("create_matrix", edit=T)

#-------------------- Reading in the data -------------------------
mbtiData <- read.csv("Data/mbti_1.csv")
mbtiData$posts <- as.character(mbtiData$posts)

lvls <- levels(mbtiData$type) # save the levels for later use

#----------------------------------- Baseline---------------------------
# To come up with a baseline for the model, I will take the most frequent
# personality type in the data. The most frequent type has an accuracy of 21%/24%.
# INFP is the most frequent one with 1832

table(mbtiData$type)
max(table(mbtiData$type))/nrow(mbtiData) # the most frequent type

# Save the table to work with it
tableMbti <- as.data.frame(table(mbtiData$type))
rownames(tableMbti) <- tableMbti$Var1

1/nrow(tableMbti) # the random probability of getting it right in general

# Distribution of the personality types across the two main dimensions
ES <- tableMbti["ESFJ",2] + tableMbti["ESFP",2] +
  tableMbti["ESTJ",2]+tableMbti["ESTP",2]

EN <- tableMbti["ENFJ",2] + tableMbti["ENFP",2] +
  tableMbti["ENTJ",2] + tableMbti["ENTP",2]

IN <- tableMbti["INFJ",2] + tableMbti["INFP",2] +
  tableMbti["INTJ",2] + tableMbti["INTP",2]

IS <- tableMbti["ISFJ",2] + tableMbti["ISFP",2] +
  tableMbti["ISTJ",2] + tableMbti["ISTP",2]

EsensingRatio <- ES/(EN+ES) #10,9%
IsensingRatio <- IS/(IN+IS) #14,7%
EIRatio <- (ES+EN)/(IS+IN+ES+EN) #23%

#------------------ A more robust model  ------------------------
# skip this part if you want to proceed with the full model

# # The entire script predicts personality types. As the people in the data
# # mention their own personality type and might therefore bias the algorithm,
# # the following excludes their own type from the raw text.
# 
# mbtiData$type <- as.character(mbtiData$type)
# 
# mbtiUnbiased <- mbtiData
# 
# # Remove own type from post
# for (i in 1:nrow(mbtiData)){
#   mbtiUnbiased[i,"posts"] <- removeWords(mbtiData[i,"posts"], mbtiData[i, "type"])
# }
# 
# mbtiData <- mbtiUnbiased # reassign it to the main data object

#--------------------- A more robust model 2-----------------------
# # skip this part if you want to proceed with the full model
# 
# # As some types are underrepresented in the data, this keeps only types
# # that occured more often than 271 times (5% of expected)
# 
# mbtiShort <- NULL
# vectorKeep <- as.character(plyr::count(mbtiData$type)[plyr::count(mbtiData$type)$freq > 271, 1])
# 
# for (i in 1:nrow(mbtiData)){
#   if (mbtiData[i, "type"] %in% vectorKeep){
#     mbtiShort <- rbind(mbtiShort, mbtiData[i,])
#   }
# }
# 
# # Adjust the lvls object as there are now less levels than before.
# mbtiShort$type <- as.factor(as.character(mbtiShort$type))
# lvls <- levels(mbtiShort$type)
# 
# mbtiData <- mbtiShort

#-------------------- Feature extraction --------------------------
# The following gathers any promising features, mostly derived from literature.

# Extraverts post more links
mbtiData$links <- mbtiFeature(words = "http", feature = "links",
                                 mbtiData = mbtiData, fullWord = FALSE)

mbtiData$youtube <- mbtiFeature(words = c("youtube", "youtu.be"), feature = "youtube",
                           mbtiData = mbtiData, fullWord = c(F, F))

# Extraverts refer more often to themselves
selfVector <- c("me", "I", "myself", "mine", "I'")
mbtiData$self <- mbtiFeature(words = selfVector, feature = "self", mbtiData = mbtiData,
                                fullWord = c(T, T, T, T, F))

mbtiData$postLength <- nchar(mbtiData$posts)

# Emojis in the text start with a ":"
mbtiData$emojis <- mbtiFeature(words = ":[:alpha:]", feature = "emoji",
                                  mbti = mbtiData, fullWord = FALSE)
mbtiData$symbols <- mbtiFeature(words = "[:punct:]", feature = "symbols",
                                   mbti = mbtiData, fullWord = FALSE)

mbtiF <- mbtiData # to save our features in a separate object


#-------------------- Linguistic features for the plain model--------------------
# Using the quanteda dictionary, some more features are extracted. These will 
# however be used in a separate SVM to the dt matrix.
mbtiCorpus <- corpus(mbtiData, text_field = "posts") # quanteda step 1: corpus
LIWCmbti <- liwcalike(mbtiCorpus,
                        dictionary = data_dictionary_NRC) # quanteda step 2: run with dict


mbtiDataLing <- cbind(mbtiData, LIWCmbti) # add features to df
mbtiDataLing <- mbtiDataLing[, - which(names(mbtiDataLing)=="docname")] # remove docname column

# This saves our new features in a separate object -> again, if this step is skipped!
mbtiFLing <- mbtiDataLing

#---------------------------- Running SVMs ------------------------------
# The following runs the main SVM, altered to test variations of the model
# specifications. The process uses svmFunction from mbti_functions.R

# RTextTools::create_analytics cannot handle a factor or a string, therefore num.
mbtiData$type <- as.numeric(as.factor(mbtiData$type))

######## All SVMs (1/2)
# Options: plain model (1) or model with options (2)
# Input: only text (a) or only features (b) or both (c)

# The underlying logic is to test for each of the lines seperately.
# documentTermMatrix <- create_matrix(mbti$posts,
#                                     minDocFreq = 3,
#                                     ngramLength=1,
#                                     removePunctuation = FALSE,
#                                     removeStopwords = FALSE,
#                                     stemWords = TRUE,
#                                     toLower = FALSE,
#                                     weighting = "tm:weightTfIdf")

# First step: a documentTermMatrix.
woSpecifications1a <- create_matrix(mbtiData$posts) # 1a)
# see below for 1b)
woSpecifications1c <- create_matrix(mbtiData[,2:ncol(mbtiData)]) # 1c)

# All specifications for 2)
wordOccurence <- create_matrix(mbtiData$posts, minDocFreq = 3)
w2gram <- create_matrix(mbtiData$posts, ngramLength=2)
w3gram<- create_matrix(mbtiData$posts, ngramLength=3)
w5gram <- create_matrix(mbtiData$posts, ngramLength=5)
wPunctuation <- create_matrix(mbtiData$posts, removePunctuation = FALSE)
wStopwords <- create_matrix(mbtiData$posts, removeStopwords = FALSE)
wStemWords <- create_matrix(mbtiData$posts, stemWords = TRUE)
capitalization <- create_matrix(mbtiData$posts, toLower = FALSE)
TfIdf <- create_matrix(mbtiData$posts, weighting = "tm:weightTfIdf")

allSVMs <- list(wordOccurence, w2gram, w3gram, w5gram,
                wPunctuation, wStopwords, wStemWords, capitalization, TfIdf,
                woSpecifications1a, woSpecifications1c)
allSVMnames <- c("wordOccurence", "w2gram", "w3gram", "w5gram",
                 "wPunctuation", "wStopwords", "wStemWords", "capitalization",
                 "TfIdf", "woSpecifications1a", "woSpecifications1c")

# Next step: Running the SVMs based on the documentTermMatrix and the features.
for (i in 1:length(allSVMs)){
  SVMname = allSVMnames[i]
  cat(paste0("Current model: ", SVMname, ".\n"))
  svmFunction(allSVMs[[i]], mbti = mbtiData, name = SVMname)
}

##### Hyperparameter tuning
# For hyperparameter tuning, we set hyperparam = TRUE in my custom svmFunction.
# We then need to call the second part of the process with the new parameters.
# This is done with woSpecifications1c as it yielded the best results.
svmFunction(woSpecifications1c, mbti = mbtiData, name = "woSpecifications1c",
            hyperparam = TRUE)
results <- trainMySvm(name = "woSpecifications1c", costs = 0.1,
                      filename = "containerFilewoSpecifications1c")

#------------------- Cleaning the results to display plots -----------
# Out of all SVMs that ran, depending on if you skipped the robustness
# steps or not, the plain woSpecifications1c performs best (or second best).
# To analyze the results and get insights into the mechanisms, we therefore
# use the woSpecifications1c model.

# You can either use the provided resultsFilewoSpecifications1c file or 
# read in your own resultsfile.
results <- readRDS(file = "SVMs/resultsFilewoSpecifications1c")
View(results) # Labels and probabilities

# The package is very confusing with providing the "correct" label! That's
# why I wrote it out myself instead of getting it from analytics.
trainData <- nrow(mbtiData)*0.8
comparison <- cbind(results, as.factor(mbtiData[(trainData+1):nrow(mbtiData),1]))

colnames(comparison) <- c("SVM_Label", "probability", "ACTUAL_LABEL")

View(comparison)

# Converting back from numeric factors to meaningful factors
levelVector <- 1:length(lvls)
levelVector <- as.character(levelVector)

comparison$SVM_Label <- mapvalues(comparison$SVM_Label, from = levelVector, to = lvls)
comparison$ACTUAL_LABEL <- mapvalues(comparison$ACTUAL_LABEL, from = levelVector,
                                     to = lvls)

# row-wise comparison
resultComparison <- matrix(NA, nrow = nrow(comparison), ncol = 1)
for (i in 1:nrow(comparison)){
  resultComparison[i] <- comparison[i,1] == comparison[i,3]
}
sum(resultComparison)/nrow(comparison) #CORRECT in test set: 62%

#---------------------- Visual analysis of the results --------------
# Confusion Matrix
confMatrix <- confusionMatrix(comparison$SVM_Label, comparison$ACTUAL_LABEL)

### Heatmap
melted_confMatrix <- melt(confMatrix$table)
ggplot(data = melted_confMatrix, aes(x=Prediction, y=Reference, fill=value)) + 
  geom_tile()+
  scale_fill_gradient2(high = "slategray", mid = "white", 
                       space = "Lab", name = "")+
  geom_text(aes(Prediction, Reference, label = value), color = "black", size = 2)+
  labs(x = "", y = "")+
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) #rotated labels

# Histogram of the original data to compare
mbtiData$type <- as.factor(mbtiData$type)
mbtiData$type <- mapvalues(mbtiData$type, from = levelVector,
          to = lvls)

ggplot(data.frame(mbtiData), aes(x=type)) +
  geom_bar(fill = "slategray4")+
  geom_text(stat='count', aes(label=..count..), #to have the values...
            position = position_dodge(0.9), vjust = 0, size=3)+ #above the bars
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+ #rotated labels
  geom_segment(aes(x = 0, xend = length(lvls)+1, y = 271, yend = 271), # add a horizontal line
               color = "pink3", linetype = 2)+
  ggtitle("Distribution of personality types within the data")+labs(x= "", y = "")


### A ROC curve
# I followed this approach:
# https://rstudio-pubs-static.s3.amazonaws.com/224325_26efd379e9984590bb0025c5ba6dd149.html
# https://pedroconcejero.wordpress.com/2016/03/07/a-roc-curves-tutorial-part-i/

svm_out2 <- results %>% mutate(SVM_PROB2 = ifelse(SVM_LABEL == 0, 1 - SVM_PROB, SVM_PROB)) 

library(pROC)

predictedLabels <- comparison$SVM_Label

### Multi-class ROC curve
# Unreadable, but technically the correct representation.
roc.multi <- multiclass.roc(comparison$ACTUAL_LABEL, as.numeric(comparison$SVM_Label))
auc(roc.multi)

rs <- roc.multi[['rocs']]
plot.roc(rs[[1]])
sapply(2:length(rs),function(i) lines.roc(rs[[i]],col=i))


### One ROC curve using the probabilities
# Readable, but simplified.
pROC::plot.roc(predictedLabels, svm_out2$SVM_PROB2,
               print.auc = TRUE, main = "ROC curve for classifiying MBTI Personalities",
               xlim = c(1,0))
# One note of caution: pROC uses other names for axes. Sensitivity = TPR, but you must
# know that x-axis is (1-Specificity) = FPR. Or also Specificity = 1 – FPR.


#-------------------- 1b) only with features --------------------
# (without options (2) as the model doesn't perform too well overall)
# As the package only works with documentterm matrices, we need to find a 
# different solution for the linguistic features added to mbtiDataLing.
# mbtiDataLing includes _all_ features, linguistic and manual ones from above,
# but not the raw text.

## Clean the data for this approach
wordcountVector <- c(3,4,5, 7, 8) # enter all columns with features here
# bring the features decoded as selfFeatureLisa (for DT matrix) back to word count
for (i in 1:nrow(mbtiDataLing)){
  for (j in wordcountVector){ # adjust the 8 if there is more than 5 of your own features!
    if (is.na(mbtiDataLing[i,j])){ # if NA, put 0
      mbtiDataLing[i,j] <- 0
    }else{
      mbtiDataLing[i,j] <- wordcount(x = mbtiDataLing[i, j]) #otherwise wordcount
    }
  }
}

for (j in 3:ncol(mbtiDataLing)){ # back to numeric
  mbtiDataLing[,j] <- as.numeric(mbtiDataLing[,j])
}

## Run the SVM only with features
trainData <- 0.8*nrow(mbtiDataLing)
featureSVM <- svm(x = mbtiDataLing[1:trainData,3:ncol(mbtiDataLing)],
                  y = mbtiDataLing[1:trainData, "type"],
                  kernel = "linear",
                  probability = TRUE) # to get a vector of probabilities
saveSmartRDS(featureSVM)

# Read in the provided featureSVM file if you don't want to run it:
# featureSVM <- readRDS(file = "SVMs/featureSVM")

summary(featureSVM)

pred <- predict(featureSVM, newdata=mbtiDataLing[(trainData+1):nrow(mbtiDataLing),
                                                 3:ncol(mbtiDataLing)],
                probability = TRUE)

head(attr(pred,"probabilities"))

#------------------- 1b) Cleaning for Visualizations ------------------------
# Cleaning
comparison <- as.data.frame(cbind(pred, pred, mbtiDataLing[(trainData+1):nrow(mbtiDataLing),1]))
colnames(comparison) <- c("SVM_Label", "probability", "ACTUAL_LABEL")

levelVector <- 1:length(lvls)
levelVector <- as.character(levelVector)
comparison$SVM_Label <- factor(comparison$SVM_Label, levels = levelVector)
comparison$ACTUAL_LABEL <- factor(comparison$ACTUAL_LABEL, levels = levelVector)

View(comparison)

# Converting back from numeric factors to meaningful factors
comparison$SVM_Label <- mapvalues(comparison$SVM_Label, from = levelVector, to = lvls)
comparison$ACTUAL_LABEL <- mapvalues(comparison$ACTUAL_LABEL, from = levelVector, to = lvls)

# row-wise comparison
resultComparison <- matrix(NA, nrow = nrow(comparison), ncol = 1)
for (i in 1:nrow(comparison)){
  resultComparison[i] <- comparison[i,1] == comparison[i,3]
}
sum(resultComparison)/nrow(comparison) 

#--------------------- 1b) Visual analysis of the results -----------------
# It seems as if the only features option is merely beating the baseline.
# Confusion Matrix
confMatrix <- confusionMatrix(comparison$SVM_Label, comparison$ACTUAL_LABEL)

# Heatmap
melted_confMatrix <- melt(confMatrix$table)
ggplot(data = melted_confMatrix, aes(x=Prediction, y=Reference, fill=value)) + 
  geom_tile()+
  scale_fill_gradient2(high = "slategray", mid = "white", 
                       space = "Lab", name = "")+
  geom_text(aes(Prediction, Reference, label = value), color = "black", size = 2)+
  labs(x = "", y = "")+
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) #rotated labels

# Histogram of the original data to compare
mbtiDataLing$type <- as.factor(mbtiDataLing$type)
ggplot(data.frame(mbtiDataLing), aes(x=type)) +
  geom_bar(fill = "slategray4")+
  geom_text(stat='count', aes(label=..count..), #to have the values...
            position = position_dodge(0.9), vjust = 0, size=3)+ #above the bars
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+ #rotated labels
  geom_segment(aes(x = 0, xend = length(lvls)+1, y = 271, yend = 271), # add a horizontal line
               color = "pink3", linetype = 2)+
  ggtitle("Distribution of personality types within the data")+labs(x= "", y = "")


#----------------- Running other models ---------------------
# Apart from SVMs, other models such as a maximum entropy model or a random
# forest.

## Read in a container we created above. Adjust the container if you like.
container <- readRDS(file = "SVMs/containerFilewoSpecifications1c")

## Run the models
# Random forest
tree_model = train_model(container, 'TREE')
saveSmartRDS(tree_model)
cross_validate(container, nfold = 5, algorithm = 'TREE')

# Maximum entropy model
maxent_model = train_model(container, 'MAXENT')
saveSmartRDS(maxent_model)
cross_validate(container, nfold = 5, algorithm = 'MAXENT') # achieves 59%


## Analyze the outcome of the models

# SVM
#svm_out = classify_model(container, svm_model)
#svm_out <- readRDS(file = "SVMs/resultsFilewoSpecifications1c")
#svm_out <- results

# Random Forest
tree_out = classify_model(container, tree_model)
saveSmartRDS(tree_out)
tree_analytics = create_analytics(container, tree_out)
summary(tree_analytics)

# Maximum Entropy
maxent_out = classify_model(container, maxent_model)
saveSmartRDS(maxent_out)
maxent_analytics = create_analytics(container, maxent_out)
summary(maxent_analytics)