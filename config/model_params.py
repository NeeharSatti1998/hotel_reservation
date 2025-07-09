from scipy.stats import randint,uniform


LIGHTGBM_PARAMS = {
    'num_leaves': randint(20, 150),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(100, 1000),
    'min_child_samples': randint(10, 50),        
    'subsample': uniform(0.6, 0.4),                   
    'colsample_bytree': uniform(0.5, 0.5)
}         


RANDOM_SEARCH_PARAMS = {
    'n_iter' : 10,
    'cv' : 5,
    'verbose' : 2,
    'n_jobs' : -1,
    'random_state' : 42,
    'scoring' : 'accuracy'
}