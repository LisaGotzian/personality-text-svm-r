#--------------- Analyzing the MBTI Personality Data - Functions ---------------
#
# This script accompanies mbti_executables.R and provides the core functions.
# It follows the approach described in here:
# https://www.svm-tutorial.com/2014/11/svm-classify-text-r/ 

# Lisa Gotzian, March 2019

# -------------------- Preliminaries -------------------
library(RTextTools)
library(e1071)
# Under the hood, RTextTools uses the e1071 package which is a R wrapper around libsvm

# As it is needed throughout the script, create new folder called "SVMs" if not
# already there.
ifelse(!dir.exists(file.path("SVMs")),
       dir.create(file.path("SVMs")), FALSE)

#------------------------ Functions -------------------------------
saveSmartRDS <- function(object){ 
  
  ## This function saves object from within R as RDS and gives it a systime tag.
  ## This way, results from earlier analyses can be retrieved.
  
  objectFile <- paste0(deparse(substitute(object)),"File", Sys.time())
  saveRDS(object, file=paste0("SVMs/",objectFile))
}


  
mbtiFeature <- function(words, mbtiData, feature, fullWord = TRUE){
  
  ## Function to denote features in the text: I used a workaround by replicating
  ## the words with "featureLisa" and adding it to the end of the text sequence
  
  featureVector <- rep(NA, nrow(mbtiData)) # first only a NA-vector to work with the grepl function
  
  for (i in 1:length(words)){ #if multiple words for one feature
    word <- words[i]
    featureLogical <- grepl(word, mbtiData$posts) # returns a logical vector if the post contains the feature
    
    # I give up doing it in vectorized form
    
    postsSplit <- c()
    for (post in 1:nrow(mbtiData)){
      if (featureLogical[post] == TRUE){
        #extracts the posts one by one instead of creating a non-handable matrix
        postsSplit <- gsub("|||", " ", mbtiData[post, "posts"], fixed = TRUE)
        postsSplit <- strsplit(postsSplit, split = " ")
        
        if (fullWord[i] == TRUE){ # if the full word is taken
          featureSum <- sum(postsSplit[[1]] == trimws(word))
        }else{
          featureSum <- sum(grepl(trimws(word), postsSplit[[1]])) # sum of feature occurences posted by person i
        }
        
        featureText <- paste0(feature, "FeatureLisa")
        # This repeats the word times the number of occurences of textdata as the documenttermmatrix
        # from tm() used in create_matrix() isn't entirely editable by hand. It's a cheap
        # workaround, but I think it makes the code clearer.
        
        if (is.na(featureVector[post] == TRUE)){ # I am a 100% certain there is an easier way for the NA thing
          featureVector[post] <- paste(rep(featureText, featureSum), sep="", collapse=" ")
        }else{
          featureVector[post] <- paste(paste(featureVector[post]), paste(rep(featureText, featureSum), sep="", collapse=" "))
        # to combine with featureVectors of earlier words
          }
        
      }
    }
  }
  return(featureVector)
}



svmFunction <- function(documentTermMatrix, mbti = mbtiData, name,
                        hyperparam = FALSE){
  
  ## Function to run the SVMs including the analysis of results
  
  cat(paste0("Welcome to another round. This is the SVM at ", Sys.time(), ".\n"))
  
  # Preliminaries
  trainData <- nrow(mbti)*0.8
  predSize <- nrow(mbti)-trainData
  
  
  ### Training the data
  container <- create_container(documentTermMatrix, mbti$type, trainSize = 1:trainData,
                                testSize = (trainData+1):nrow(mbti),
                                virgin= FALSE)
  saveSmartRDS(container)
  
  
  # Tuning hyperparameters following this approach: https://www.jeremyjordan.me/hyperparameter-tuning/
  if(hyperparam == TRUE){
    ###### IF TUNING then end the function here and adjust the results
    cat(paste0("Tuning it...\n"))

    # tune.svm() is a convenience tuning wrapper function
    model.tuned <- tune.svm(x = container@training_matrix,
                            y = container@training_codes,
                            kernel = "linear", # different parameters
                            cost = 10^(-1:1)
                            # fill in any other SVM params as needed here
    )

    return(model.tuned)
    # Careful! You will now need to call trainMySvm separately to enter the hyperparams.
    
    # Or continue without parameter tuning
  }else{
    results <- trainMySvm(name = name, container = container,
                          fromhyperparam = FALSE)
    return(results)
  }
}
  




trainMySvm <- function(name, cost = 0.1, filename = NULL, #enter hyperparameters here
                       container = NULL, fromhyperparam = TRUE){ 
  
  ## This is the subsequent function to svmFunction and only needs to be
  ## called manually if hyperparamters were tuned.
  
  cat(paste0("Training it...\n"))
  
  if(fromhyperparam == TRUE){
    container <- readRDS(file = paste0("SVMs/", filename)) #only if hyperparameter
  }
  
  # linear model because text is often linearly separable (Joachims)
  model <- train_model(container, algorithm="SVM", kernel="linear", cost = 0.1) # enter hyperparameters here
  saveSmartRDS(model)
  
  ### Testing the model
  
  cat(paste0("Giving you results...\n"))
  results <- classify_model(container, model)
  saveSmartRDS(results)
  
  cat(paste0("Analyzing the results...\n"))
  analytics <- create_analytics(container, results)
  
  # And save them to my working directory.
  write.table(summary(analytics),
              file = paste0("SVMs/analysis/analysis", name))
  
  cat(paste0("###########################################################################\n\n"))
  
  saveSmartRDS(analytics)
  
  # What do the different analytics mean?
  # analytics@algorithm_summary: SUMMARY OF PRECISION, RECALL, F-SCORES, AND ACCURACY SORTED BY TOPIC CODE FOR EACH ALGORITHM
  # analytics@label_summary: SUMMARY OF LABEL (e.g. TOPIC) ACCURACY
  # analytics@document_summary: RAW SUMMARY OF ALL DATA AND SCORING
  # analytics@ensemble_summary: SUMMARY OF ENSEMBLE PRECISION/COVERAGE. USES THE n VARIABLE PASSED INTO create_analytics()
  
  # analytics@ensemble_summary
  
  return(results) #I only return this for the plot function
}