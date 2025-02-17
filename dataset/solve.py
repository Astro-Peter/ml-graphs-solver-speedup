from pyscipopt import Model
from random import randint
from solution import Solution


def solve(A, c, b, time_constraint, total_solutions, vtype="I", objective_task="maximize"):
    """
    Compute either total_solutions solutions, or compute solutions
    until solver runtime is >= time_constraint
    
    Keyword arguments:
        A -- constraint matrix of size (m, n)
        c -- vector of objective matrix coefficients of size n
        b -- right-hand side vector of size m
        time_constraint -- time given for solver to run in seconds
        total_solutions -- number of solutions required
        vtype -- the type of solution required
        objective_task -- what to do with the objective function
    Returns: 
        a generator, returning Solution class
    """
    solutions = []
    cnt = 0
    for i in range(total_solutions):
        model = Model()
        model.hideOutput(True)
        model.setParam("limits/time", time_constraint)
        n = len(c)
        m = len(b)
        vars = [model.addVar(vtype=vtype, name=f"x_{i}") for i in range(n)]
        for i in range(m):
            model.addCons(sum(A[i][j] * vars[j] for j in range(n)) <= b[i])
        for x in solutions:  
            model.addCons(
                sum(abs(x[i] - vars[i]) for i in range(n)) >= 1
            )
        model.setObjective(sum(c[i] * vars[i] for i in range(n)), objective_task)
        random_state = randint(0, 2147483647)
        model.setParam("randomization/randomseedshift", random_state)
        model.optimize()
        if model.getStatus() != "optimal":
            break
        x = [model.getVal(vars[i]) for i in range(n)]
        solutions.append(x)
        obj_fun_val = sum([x[i] * c[i] for i in range(n)])
        runtime = model.getSolvingTime()
        yield Solution(x, cnt, runtime, obj_fun_val, random_state)
        cnt += 1


# example of solver working
if __name__ == "__main__":
    A = [[1, 2, 5],
         [2, 4, 7]]
    b = [20, 20]
    c = [1, 8, 4]
    solutions = solve(A, c, b, 10, 1000)
    for solution in solutions:
        print(solution.x, solution.solve_time, solution.target_val)
        print(solution.index)