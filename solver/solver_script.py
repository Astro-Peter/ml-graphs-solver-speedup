import argparse
import os
import csv
import numpy as np
from pyscipopt import Model
from solve import solve

def main():
    parser = argparse.ArgumentParser(description='Find multiple solutions for lp problems.')
    parser.add_argument('input_dir', help='Directory containing .lp files')
    parser.add_argument('time_limit', type=int, help='Time limit in seconds per problem')
    parser.add_argument('n_solutions', type=int, help='Number of solutions to find')
    parser.add_argument('-o','--output_dir', default='solutions', help='Output directory for solutions')
    parser.add_argument('-c', '--csv_file', default='solutions/solutions.csv', help='CSV file name')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare CSV file
    csv_exists = os.path.isfile(args.csv_file)
    with open(args.csv_file, 'a', newline='') as csvfile:
        fieldnames = ['lp_file', 'solution_file', 'random_states_file', 'solve_times_file']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not csv_exists:
            writer.writeheader()

        # Process each .lp file in input directory
        for lp_file in os.listdir(args.input_dir):
            if not lp_file.endswith('.lp'):
                continue

            lp_path = os.path.join(args.input_dir, lp_file)
            filename = os.path.basename(lp_file)
            
            try:
                model = Model()
                model.readProblem(lp_path)
                model.setParam('limits/time', args.time_limit)
                model.hideOutput()
                solution_file = f"{args.output_dir}/{filename}_solution.npy"
                random_states_file = f"{args.output_dir}/{filename}_random_states.npy"
                solve_times_file = f"{args.output_dir}/{filename}_solve_times.npy"


                solutions = solve(args.n_solutions, model)
                solutions_arr = [] # arr of solutions to the problem, later converted to a .npy file
                random_states = [] # arr of random states, has the shape (n) 
                solve_times = [] # arr of solve times, has the shape (n)
                for solution in solutions:
                    if solution.status != "optimal":
                        break
                    
                    solutions_arr.append(solution.solution.x)
                    random_states.append(solution.solution.random_state)
                    solve_times.append(solution.solution.solve_time)

                np.save(solution_file, np.array(solutions_arr))
                np.save(random_states_file, np.array(random_states))
                np.save(solve_times_file, np.array(solve_times))
                writer.writerow(
                    {
                        'lp_file': lp_path,
                        'solution_file': solution_file,
                        'random_states_file': random_states_file,
                        'solve_times_file': solve_times_file
                    }
                )


            except Exception as e:
                print(f"Error processing {lp_file}: {str(e)}")
                continue

if __name__ == '__main__':
    main()