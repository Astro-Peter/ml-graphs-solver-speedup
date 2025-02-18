from solution import Solution

class SolutionResultType:
    solution: Solution
    status: str

    def __init__(self, status: str, solution: Solution = None):
        self.status = status
        self.solution = solution