import numpy as np
from scipy import optimize
import Json as js
import sys
from itertools import chain, combinations
from scipy.spatial.distance import hamming

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Process-JSON.py <data.json>")
        exit()

    data_file = sys.argv[1]
    biology = js.get_data(data_file)

    # first read in a
    # c is the constraint
    # we want to trick the linear solver into calculating the absolute values.

    # We have a set of function coefficients characterised by Boolean patterns of the same length as the bit patterns of points in the space.
    # We want to work out the values of points using the appropriate functions.  Functions are appropriate to a point if their True positions are a subset of the one positions of the point.
    # So each point gives us a set of function coefficients
    # Now the values for each pair of points can be compared by size.  This gives an inquality (or equality) for the two sets of cofficients corresponding to the two points.
    # We will use a dictionary mapping points to lists of coefficients

    # There are way more efficient ways to build this list.
    coeffs = {}
    size = biology[0]
    data = biology[2]
    coords = len(biology[1])
    for dt in data:
        ones = np.nonzero(dt[0])[0]  # Where it is a 1
        coeffs[dt[0]] = np.full(size, 0)
        subsets = powerset(ones)
        for l in subsets:
            v = [('1' if i in l else '0') for i in range(coords)]
            str_v = "".join(v)
            coeffs[dt[0]][int(str_v,2)] = 1
        print("Point = ", dt[0], "Appropriate Coefficients are: ", coeffs[dt[0]])

    equals_constraints = np.full(size, 0)
    inequality_constraints = np.full(size, 0)

    for first in range(size):
        for second in range(first + 1, size):
            # The next if checks to see that we are just doing the fitness graph edges.
            if coords * hamming(data[first][0], data[second][0]) != 1:
                continue
            inequality = np.array(coeffs[data[first][0]]) - np.array(coeffs[data[second][0]])
            if data[first][1] < data[second][1]:
                inequality = -1 * inequality
            if data[first][1] == data[second][1]:
                equals_constraints = np.vstack((equals_constraints, inequality))
            else:
                inequality_constraints = np.vstack((inequality_constraints, inequality))
                print(data[first][0], "(", "{:.2f}".format(data[first][1]), ")", data[second][0], "(", "{:.2f}".format(data[second][1]), ")", inequality)

    c = np.full(size, 1)
    inequality_constraints = np.delete(inequality_constraints, 0, 0)
    equals_constraints = np.delete(equals_constraints, 0, 0)
    ineq_bounds = np.full(np.ma.size(inequality_constraints, 0),-1)
    eq_bounds = np.full(np.ma.size(equals_constraints, 0),0)
    print(np.ma.size(inequality_constraints, 0), np.ma.size(equals_constraints, 0))
    boundx = [(-100, 100) for x in range(size)]
    boundx[0] = (0,0)


    result = optimize.linprog(c, A_ub=inequality_constraints, b_ub=ineq_bounds, A_eq=equals_constraints, b_eq=eq_bounds, bounds=boundx, options={'sym_pos':False,'cholesky':False,"presolve":False}, method='interior-point')
    print(np.around(result['x'], 1), result['message'])

    # Now we add in extra variables and use these to minimise the absolute values of the coefficients.
    # We need as many fake variables as original ones.
    # Each one has two constraints which make sure that it is as least as big as x_i and -x_i - so ...-1.....-1...  and ...1....-1...
    #                                                                                              ....i.....i+n... and ...i...i+n....
    # Plus all original rows.
    extra_inequalities = 2 * size
    extra_vars = size
    inequality_constraints = np.pad(inequality_constraints, ((0, extra_inequalities), (0, extra_vars)), mode='constant', constant_values=0)
    equals_constraints = np.pad(equals_constraints, ((0, 0), (0, extra_vars)), mode='constant', constant_values=0)
    ineq_bounds = np.pad(ineq_bounds, ((0,extra_inequalities)), mode='constant', constant_values=0)

    for index in range(extra_vars):
        inequality_constraints[-extra_vars + index][index] = -1
        inequality_constraints[-extra_vars + index][extra_vars + index] = -1
        inequality_constraints[-2 * extra_vars + index][index] = 1
        inequality_constraints[-2 * extra_vars + index][extra_vars + index] = -1
    c = np.pad(c, ((extra_vars, 0)),  mode='constant', constant_values=0)
    boundx = boundx + list((0,100) for x in range(extra_vars))

    result = optimize.linprog(c, A_ub=inequality_constraints, b_ub=ineq_bounds, A_eq=equals_constraints, b_eq=eq_bounds, bounds=boundx, options={'sym_pos':False,'cholesky':False,"presolve":False}, method='interior-point')

    coefficients = ( list(np.nonzero(list(map(int,bin(x)[2:].zfill(coords))))[0]) for x in range(size))
    print ( "\n".join(map(lambda x: str(x[0]) + ": " + str(x[1]), zip(coefficients, np.around(result['x'][0:size], 1)))))
    #print (list(map(lambda x: ":".join(x), zip(answer, np.around(result['x'][0:size], 1))))
    print(np.around(result['x'][0:size], 1))
    print(np.around(result['x'][-size:], 1))
    print(result['message'])
    exit(0)