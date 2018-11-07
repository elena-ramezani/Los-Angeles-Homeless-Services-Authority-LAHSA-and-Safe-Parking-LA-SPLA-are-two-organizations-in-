import random
import numpy as np
import pdb
import time

## matrix columns, all people, rows: days in a week, [m, tu, w, th, f, sa, su]
## shape = [7, no_ppl]
global days_ppl_matrix
global no_ppl
global id_list_str
global no_days
global max_parking_capacity

file_name = 'input2.txt'
output_name = 'output.txt'

def read_input(file_name):
    # id_list, map ids to person : same length as len(days_ppl_matrix[0])
    # ppl_in_both: zero one list, shows if the person is valid for both SPLA and LAHSA: same length as len(days_ppl_matrix[0])

    id_ppl_LAHSA = []
    id_ppl_SPLA = []
    ppl_inf = []
    SPLA_ppl = []
    LAHSA_ppl = []


    with open(file_name, 'r') as f:
        inp = f.readlines()
    inp = [x.replace('\n', '') for x in inp]
    i = 1

    max_parking_capacity = np.ones(7)
    max_parking_capacity = int(inp[i]) * max_parking_capacity
    i = i+1
    number_applicants_by_LAHSA = int(inp[i])
    for i in range(3, 3 + number_applicants_by_LAHSA):
        id_ppl_LAHSA.append(inp[i])
    i = i + 1
    ii = i
    number_applicants_by_SPLA = int(inp[i])
    for ii in range(i+1, i + 1 + number_applicants_by_SPLA):
        id_ppl_SPLA.append(inp[ii])
    ii = ii + 1
    total_no_ppl = int(inp[ii])
    for iii in range(ii + 1, ii + 1 + total_no_ppl):
        ppl_inf.append(inp[iii])

    ## calculate max each day capacity for SPLA
    already_assigned = [person[-7:] for person in ppl_inf if person[:5] in id_ppl_SPLA]
    days_ppl_matrix = [[int(i) for i in days] for days in already_assigned]
    days_ppl_matrix_total = np.sum(days_ppl_matrix,axis=0)
    max_parking_capacity = max_parking_capacity - days_ppl_matrix_total

    ## remainign ppl
    remaining_ppl = [ppl for ppl in ppl_inf if (ppl[:5] not in id_ppl_LAHSA and ppl[:5] not in id_ppl_SPLA)]

    ## assign to SPLA or LAHSA
    for person in remaining_ppl:
        if person[10:13].lower() == 'nyy':
            SPLA_ppl.append(person)
        if person[5].lower() == 'f' and int(person[6:9]) > 17 and person[9].lower() == 'n':
            LAHSA_ppl.append(person)

    ppl_common = list(set(SPLA_ppl).intersection(LAHSA_ppl))

    id_list_str = [ppl[:5] for ppl in SPLA_ppl]

    ppl_in_both = np.zeros(len(id_list_str))
    ind = [id_list_str.index(ppl[:5]) for ppl in ppl_common]
    ppl_in_both[ind] = 1

    days_ppl_matrix = [[int(i) for i in ppl[-7:]] for ppl in SPLA_ppl]
    days_ppl_matrix = np.array(days_ppl_matrix)
    days_ppl_matrix = days_ppl_matrix.transpose()

    no_ppl = len(days_ppl_matrix[0])

    return days_ppl_matrix, id_list_str, ppl_in_both, max_parking_capacity, no_ppl

def select_output(genetic_output, id_list_str, ppl_in_both, days_ppl_matrix):
    ## select best person form the results for genetic algorithm
    ## id_list, string list of ids
    gen_res_in_both = np.multiply(genetic_output, ppl_in_both)
    if np.sum(gen_res_in_both) != 0:
        gen_wk = np.repeat([gen_res_in_both, ], no_days, axis=0)

    else:
        gen_wk = np.repeat([genetic_output, ], no_days, axis=0)

    obj_ = np.sum(np.multiply(gen_wk, days_ppl_matrix), axis=0)
    if np.sum(obj_) == 0:
        return

    ppl_with_max_days = np.argwhere(obj_ == np.max(obj_))
    ppl_with_max_days = np.reshape(ppl_with_max_days, (ppl_with_max_days.shape[0],))
    id_list_str = np.array(id_list_str)
    id_list_int = [int(x) for x in id_list_str]
    id_list_int = np.array(id_list_int)
    out = id_list_str[np.argwhere(id_list_int == np.min(id_list_int[ppl_with_max_days]))]
    out = np.reshape(out, (out.shape[0],))
    return out.tolist()



##
def generate_population(no_population, no_ppl):
    ## matrix with number of rows: no_ppl, number of columns:no_population
    return np.random.randint(2, size=(no_population, no_ppl))


def cross_over(population_matrix):
    ## call cross_over_sub for every parent pair randomly
    index_matrix = np.random.permutation(no_population)
    for i in range(0, len(index_matrix), 2):
        population_matrix[index_matrix[i]], population_matrix[index_matrix[i + 1]] = cross_over_sub(
            population_matrix[index_matrix[i]], population_matrix[index_matrix[i + 1]])

    return population_matrix


def cross_over_sub(chromx, chromy):
    ## get two parents
    cross_over_point = np.random.randint(no_ppl)
    child1 = np.append(chromx[:cross_over_point], chromy[cross_over_point:])
    child2 = np.append(chromy[:cross_over_point], chromx[cross_over_point:])

    ## return children
    return child1, child2

def mutation(population_matrix, mutation_rate):
    row_no = np.random.randint(no_population, size=mutation_rate)
    col_no = np.random.randint(no_ppl, size=mutation_rate)
    population_matrix[row_no, col_no] = 1 - population_matrix[row_no, col_no]
    return population_matrix

def select_parents(population_matrix, fitness):
    idx = np.random.choice(np.arange(no_population), size=no_population -1 , replace=True, p=np.divide(np.array(fitness,dtype=float), sum(fitness)))
    return population_matrix[np.append(idx, np.argmax(fitness)) ] ## select best parents based on objective function


def objective_function_chrom(chrom, days_ppl_matrix, no_days, max_parking_capacity):
    ## get one chrom and calculate objective function

    chrom_wk = np.repeat([chrom, ], no_days, axis=0)
    obj_ = np.sum(np.multiply(chrom_wk, days_ppl_matrix), axis=1)
    if np.any(obj_ > max_parking_capacity):
        objective_value = 0
    else:
        objective_value = np.sum(obj_)
    return objective_value


def objective_function(population_matrix, days_ppl_matrix, no_days, max_parking_capacity):
    objective_all = []
    for chrom in population_matrix:
        objective_all.append(objective_function_chrom(chrom, days_ppl_matrix, no_days, max_parking_capacity))
    return objective_all

def print_pretty(matrix):
    for i in range(len(matrix)):
        print(matrix[i])


def genetic_algorithm():

    itere = 0
    max_fit = []
    timeout = time.time() + 170  # 3 minutes from now
    population_matrix = generate_population(no_population, no_ppl)
    while True:
        itere = itere + 1

        # pdb.set_trace()
        fitness = objective_function(population_matrix, days_ppl_matrix, no_days, max_parking_capacity)
        best_so_far = population_matrix[np.argmax(fitness)]
        worst_so_far = np.argmin(fitness)

        print('iteration: ' + str(itere) + ' best objective function:' + str(max(fitness)))
        max_fit.append(max(fitness))

        ## less than 3 min
        if time.time() > timeout or (itere > 100 and np.average(max_fit[-40:]) == max_fit[-1]):
            #pdb.set_trace()
            if sum(fitness) == 0:
                return
            winner = np.argwhere(fitness == np.max(fitness))
            unique_rows = np.unique(population_matrix[np.reshape(winner, [-1, ])], axis=0)
            return unique_rows.flatten()

        population_matrix = select_parents(population_matrix, fitness)
        population_matrix = cross_over(population_matrix)
        population_matrix = mutation(population_matrix, mutation_rate)
        population_matrix[worst_so_far] = best_so_far


    if sum(fitness) == 0:
        return
    winner = np.argwhere(fitness == np.max(fitness))
    unique_rows = np.unique(population_matrix[np.reshape(winner, [-1, ])], axis=0)
    return unique_rows.flatten()


if __name__== "__main__":
    no_days = 7

    days_ppl_matrix, id_list_str, ppl_in_both, max_parking_capacity, no_ppl = read_input(file_name)

    no_population = max(2 * no_ppl, 500)  ## even number
    mutation_rate = int(no_population * 0.04)

    best_res = genetic_algorithm()

    print(best_res)
    id_out = []
    if np.any(best_res):
        id_out = select_output(best_res, id_list_str, ppl_in_both, days_ppl_matrix)
        print(id_out)
        with open(output_name, 'w') as f:
            f.write(id_out[0])
    else:
        with open(output_name, 'w') as f:
            f.write('\n')

