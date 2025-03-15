from pyscipopt import Model
from random import randint
from result_type import SolutionResultType
from solution import Solution

def solve_from_matrices(total_solutions,  time_constraint, A, c, b, vtype="I", objective_task="maximize"):
    model = Model()
    model.hideOutput(True)
    n = len(c)
    m = len(b)
    vars = [model.addVar(vtype=vtype, name=f"x_{i}") for i in range(n)]
    model.setParam("limits/time", time_constraint)
    for i in range(m):
        model.addCons(sum(A[i][j] * vars[j] for j in range(n)) <= b[i])
    return solve(total_solutions=total_solutions, objective_task=objective_task, orig_model=model)


def solve(total_solutions, orig_model, objective_task="maximize"):
    """
    Compute either total_solutions solutions, or compute solutions
    until solver runtime is >= time_constraint
    Takes in either a scip model, or parameters for MILP problem
    If the model is given ignores any parameters for MILP problem
    If neither are given(or not full parameter list) raises an 
    exception
    
    Keyword arguments:
        orig_model -- scip model
        A -- constraint matrix of size (m, n)
        c -- vector of objective matrix coefficients of size n
        b -- right-hand side vector of size m
        time_constraint -- time given for solver to run in seconds
        total_solutions -- number of solutions required
        vtype -- the type of solution required
        objective_task -- what to do with the objective function
    Returns: 
        a generator, returning SolutionResultType class containing 
        solver status and the Solution class, describing the found
        solution, if solver's status was optimal
    """
    solutions = []
    cnt = 0
    for i in range(total_solutions):
        model = Model(sourceModel=orig_model)
        vars = model.getVars()
        n = len(vars)
        for x in solutions:
            model.addCons(
                sum(abs(x[i] - vars[i]) for i in range(n)) >= 1
            )
        model.setObjective(sum(c[i] * vars[i] for i in range(n)), objective_task)
        random_state = randint(0, 2147483647)
        model.setParam("randomization/randomseedshift", random_state)
        model.optimize()
        if model.getStatus() != "optimal":
            yield SolutionResultType(model.getStatus())
            break
        x = [model.getVal(vars[i]) for i in range(n)]
        solutions.append(x)
        obj_fun_val = sum([x[i] * c[i] for i in range(n)])
        runtime = model.getSolvingTime()
        yield SolutionResultType(model.getStatus(), Solution(x, cnt, runtime, obj_fun_val, random_state))
        cnt += 1


# example of solver working
if __name__ == "__main__":
    A = [[1, 2, 5],
         [2, 4, 7]]
    b = [20, 20]
    c = [1, 8, 4]
    solutions = solve_from_matrices(A=A, c=c, b=b, time_constraint=10, total_solutions=1000)
    for solution in solutions:
        if solution.status != "optimal":
            print(solution.status)
            continue
        solution = solution.solution
        print(solution.x, solution.solve_time, solution.target_val)
        print(solution.index)