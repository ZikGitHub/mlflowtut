from mlflow_utils import create_dataset
from hyperopt import fmin, tpe, hp, Trials

def objective_function(params):

    y = (params["x"] / 100 + 17) ** 2 + 2
    return y

search_space = {   
     "x":hp.uniform("x", -100, 100)
}

trials = Trials()

best = fmin(
    fn=objective_function,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

print(best)