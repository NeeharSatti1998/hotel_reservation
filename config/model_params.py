from scipy.stats import randint,uniform


LIGHTGBM_PARAMS = {
    'num_leaves': randint(20, 100),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.03, 0.07),
    'n_estimators': randint(100, 200),
    'min_child_samples': randint(20, 50),        
    'subsample': uniform(0.7, 0.2),                   
    'colsample_bytree': uniform(0.7, 0.3)
}         


RANDOM_SEARCH_PARAMS = {
    'n_iter' : 5,
    'cv' : 3,
    'verbose' : 2,
    'n_jobs' : -1,
    'random_state' : 42,
    'scoring' : 'accuracy'
}