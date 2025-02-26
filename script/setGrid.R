
set_getpi = function(key_df) {
    train_control <- caret::trainControl(
    method = "cv",                   
    number = 5,                      
    search = "grid",               
    classProbs = TRUE,   
    allowParallel = TRUE,  
    verboseIter = FALSE,          
    sampling = 'up',
    summaryFunction = multiClassSummary          
    )

    tune_grid <- expand.grid(
    mtry = seq(floor(sqrt(ncol(key_df))), ncol(key_df), by = 3),
    splitrule = c('gini'),
    min.node.size = c(5, 15, 25)
    )
    return(list(train_control = train_control, tune_grid = tune_grid))
}


set_gettransition = function(design_mat) {
    train_control <- caret::trainControl(
    method = "cv",
    number = 5,
    search = "grid",
    verboseIter = FALSE,
    allowParallel = TRUE,
    classProbs = TRUE,
    summaryFunction = multiClassSummary
    )

    tune_grid <- expand.grid(
    mtry = seq(floor(sqrt(ncol(design_mat) - 1)), ncol(design_mat) - 1, 2),
    splitrule = c('gini'),
    min.node.size = c(5, 15, 25)
    )
    return(list(train_control = train_control, tune_grid = tune_grid))
}


set_getcondY = function(design_mat) {
    tune_control <- caret::trainControl(
    method = 'cv', 
    number = 5,
    search = 'grid', 
    verboseIter = FALSE, 
    allowParallel = TRUE
    )

    tune_grid <- expand.grid(
    mtry = seq(floor(sqrt(ncol(design_mat) - 1)), ncol(design_mat) - 1, 2),
    splitrule = c('variance'),
    min.node.size = c(5, 15, 25)
    )
    return(list(tune_control = tune_control, tune_grid = tune_grid))
}


get_getcondB2 = function(covars) {
    train_control <- caret::trainControl(
        method = "cv", 
        number = 5, 
        search = "grid",
        allowParallel = TRUE, 
        verboseIter = FALSE, 
        classProbs = TRUE, 
        summaryFunction = multiClassSummary
    )
    
    tune_grid <- expand.grid(
    mtry = seq(floor(sqrt(ncol(covars))), min(floor(sqrt(ncol(covars))) + 6, ncol(covars)), 3), 
    min.node.size = c(5, 15, 25), 
    splitrule = 'gini'
    )
    return(list(train_control = train_control, tune_grid = tune_grid))
}