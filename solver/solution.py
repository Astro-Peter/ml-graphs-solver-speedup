class Solution:
    x: list # MILP solution
    index: int # the number of this solution
    solve_time: float # solver runtime in seconds
    random_state: int # the random state used for 
    # the solver when this solution was found

    def __init__(self, x: list, index: int, solve_time: float, random_state: int):
        self.x = x
        self.index = index
        self.solve_time = solve_time
        self.random_state = random_state