#%%
import ray
ray.shutdown()
ray.init(num_cpus=4)

import matplotlib.pyplot as plt
from deap import gp

#from dctkit import config
#from dctkit.math.opt import optctrl as oc
from alpine.gp import gpsymbreg as gps
from alpine.data import Dataset
from alpine.gp import util
import numpy as np

import yaml
import os

# import jax.numpy as jnp
import time

# from jax import jit, grad
import warnings
import pygmo as pg

# from functools import partial
import re
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#from datasets import generate_dataset

print("Done importing")
num_cpus = 4




#%%
# config()


def check_trig_fn(ind):
    return len(re.findall("cos", str(ind))) + len(re.findall("sin", str(ind)))


def check_nested_trig_fn(ind):
    return util.detect_nested_trigonometric_functions(str(ind))


def eval_model(individual, D, consts=[]):
    warnings.filterwarnings("ignore")
    y_pred = individual(*D.X, consts)
    return y_pred


def compute_MSE(individual, D, consts=[]):
    y_pred = eval_model(individual, D, consts)
    MSE = np.mean((D.y - y_pred) ** 2)

    if np.isnan(MSE) or np.isinf(MSE):
        MSE = 1e8

    return MSE


# Compiles and finds the number of constants to update in each tree
def compile_individual_with_consts(tree, toolbox, special_term_name="a"):
    const_idx = 0
    tree_clone = toolbox.clone(tree)
    for i, node in enumerate(tree_clone):
        if isinstance(node, gp.Terminal) and node.name[0:3] != "ARG":
            if node.name == special_term_name:
                new_node_name = special_term_name + "[" + str(const_idx) + "]"
                tree_clone[i] = gp.Terminal(new_node_name, True, float)
                const_idx += 1

    individual = toolbox.compile(expr=tree_clone, extra_args=[special_term_name])
    return individual, const_idx


# evaluate trees using MSE and tune constats in tree using Pygmo (genetic algoithm)
def eval_MSE_and_tune_constants(tree, toolbox, D):
    individual, num_consts = compile_individual_with_consts(tree, toolbox)

    if num_consts > 0:
        
        # config()

        def eval_MSE(consts):
            warnings.filterwarnings("ignore")
            y_pred = individual(*D.X, consts)
            total_err = np.mean((D.y - y_pred) ** 2)

            return total_err

        objective = eval_MSE

        x0 = np.ones(num_consts)

        class fitting_problem:
            def fitness(self, x):
                total_err = objective(x)
                return [total_err]

            
            def get_bounds(self):
                return (-5.0 * np.ones(num_consts), 5.0 * np.ones(num_consts))

        
        # PYGMO SOLVER
        prb = pg.problem(fitting_problem())
        # algo = pg.algorithm(pg.nlopt(solver="lbfgs"))
        # algo.extract(pg.nlopt).maxeval = 10
        # algo = pg.algorithm(pg.cmaes(gen=70))
        algo = pg.algorithm(pg.pso(gen=10))
        # algo = pg.algorithm(pg.sea(gen=70))
        pop = pg.population(prb, size=70)
        # pop = pg.population(prb, size=1)
        pop.push_back(x0)
        pop = algo.evolve(pop)
        MSE = pop.champion_f[0]
        consts = pop.champion_x
        # print(pop.problem.get_fevals())
        if np.isinf(MSE) or np.isnan(MSE):
            MSE = 1e8
    else:
        MSE = compute_MSE(individual, D)
        consts = []
    return MSE, consts


def get_features_batch(
    individuals_str_batch,
    individ_feature_extractors=[len, check_nested_trig_fn, check_trig_fn],
):
    features_batch = [
        [fe(i) for i in individuals_str_batch] for fe in individ_feature_extractors
    ]

    individ_length = features_batch[0]
    nested_trigs = features_batch[1]
    num_trigs = features_batch[2]
    return individ_length, nested_trigs, num_trigs


@ray.remote(num_cpus=num_cpus)
def predict(individuals_str_batch, toolbox, dataset, penalty, fitness_scale):

    predictions = [None] * len(individuals_str_batch)

    for i, tree in enumerate(individuals_str_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        predictions[i] = eval_model(callable, dataset, consts=tree.consts)

    return predictions


@ray.remote(num_cpus=num_cpus)
def compute_MSEs(individuals_str_batch, toolbox, dataset, penalty, fitness_scale):

    total_errs = [None] * len(individuals_str_batch)

    for i, tree in enumerate(individuals_str_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        total_errs[i] = compute_MSE(callable, dataset, consts=tree.consts)

    return total_errs


@ray.remote(num_cpus=num_cpus)
def compute_attributes(individuals_str_batch, toolbox, dataset, penalty, fitness_scale):

    attributes = [None] * len(individuals_str_batch)

    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_str_batch)

    for i, tree in enumerate(individuals_str_batch):

        # Tarpeian selection
        if individ_length[i] >= 50:
            consts = None
            fitness = (1e8,)
        else:
            MSE, consts = eval_MSE_and_tune_constants(tree, toolbox, dataset)
            fitness = (
                fitness_scale
                * (
                    MSE
                    + 100000 * nested_trigs[i]
                    + 0.0 * num_trigs[i]
                    + penalty["reg_param"] * individ_length[i]
                ),
            )
        attributes[i] = {"consts": consts, "fitness": fitness}
    return attributes


def assign_attributes(individuals, attributes):
    for ind, attr in zip(individuals, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]


def alpine_test_with_consts(x,y):
    # Import .yaml
    yamlfile = "alpine_test_with_constants.yaml"
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)
    
    
    X_train, X_test, y_train, y_test = train_test_split(x,y)
    #print(X_train.shape)
    #print(y_train.shape)
   
    pset = gp.PrimitiveSetTyped("Main", [float], float)
    pset.renameArguments(ARG0="x")
    pset.addTerminal(object, float, "a") # REMEMBER THIS
   

    batch_size = 1000
    callback_func = assign_attributes
    fitness_scale = 1.0

    penalty = config_file_data["gp"]["penalty"]
    common_params = {"penalty": penalty, "fitness_scale": fitness_scale}
    

    train_data = Dataset("dataset", X_train, y_train)
    test_data = Dataset("dataset", X_test, y_test)
    #print(test_data.X.shape)
    #print(test_data.y.shape)
    
    train_data.X = [train_data.X]
    test_data.X = [test_data.X]


    seed = [
        "add(x,mul(x,add(x,mul(x,add(x,mul(x,x))))))"]  # x**3 + x**2 + x
    
    gpsr = gps.GPSymbolicRegressor(
        pset=pset,
        fitness=compute_attributes.remote,
        predict_func=predict.remote,
        error_metric=compute_MSEs.remote,
        common_data=common_params,
        callback_func=callback_func,
        print_log=True,
        num_best_inds_str=1,
        config_file_data=config_file_data,
        save_best_individual=True,
        output_path="./",
        seed=seed,
        plot_best_individual_tree = 1,
        batch_size=batch_size,
    )

    

    tic = time.time()
    gpsr.fit(train_data)
    toc = time.time()

    if hasattr(gpsr.best, "consts"):
        print("Best parameters = ", gpsr.best.consts)

    print("Elapsed time = ", toc - tic)
    time_per_individual = (toc - tic) / (
        gpsr.NGEN * gpsr.NINDIVIDUALS * gpsr.num_islands
    )
    individuals_per_sec = 1 / time_per_individual
    print("Time per individual = ", time_per_individual)
    print("Individuals per sec = ", individuals_per_sec)

    u_best = gpsr.predict(test_data)
    #print(u_best)
    #print(y_test)

    plt.figure()
    plt.plot(u_best, label='ubest')
    plt.plot(y_test, "+",label='y_test')
    plt.legend()
    plt.show()

    # show function
    data_show = Dataset("dataset", x, y)
    data_show.X = [data_show.X]

    #print(realdata.X)
    #print(realdata.y)
    u_show = gpsr.predict(data_show)
    plt.figure()
    plt.plot(x,u_show,label='model')
    plt.plot(x,y,label='data')
    plt.grid()
    plt.legend()
    plt.show()


    MSE = np.sum((u_best - y_test) ** 2) / len(u_best)
    r2 = r2_score(y_test, u_best)
    print("MSE on the test set = ", MSE)
    print("R^2 on the test set = ", r2)
    if MSE <= 1e-10:
        return 1.0
    else:
        return 0.0


#%%

#num_cpus = 4
#num_runs = 10  # 20 
 
# generate training and test datasets
x = np.array([x/10. for x in np.arange(1, 10,0.2)])
y = (np.sin(x) + np.log(x + 1) * np.exp(x)) / (x**2 - np.exp(x)) + 3*x - 2

#%%
alpine_test_with_consts(x,y)



