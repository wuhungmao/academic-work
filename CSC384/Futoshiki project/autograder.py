import itertools
import cspbase

from propagators import *
from futoshiki_csp import *

#Now n-Queens example
def queensCheck(qi, qj, i, j):
    '''Return true if i and j can be assigned to the queen in row qi and row qj 
       respectively. Used to find satisfying tuples.
    '''
    return i != j and abs(i-j) != abs(qi-qj)

def nQueens(n):
    '''Return an n-queens CSP'''
    i = 0
    dom = []
    for i in range(n):
        dom.append(i+1)

    vars = []
    for i in dom:
        vars.append(Variable('Q{}'.format(i), dom))

    cons = []    
    for qi in range(len(dom)):
        for qj in range(qi+1, len(dom)):
            con = Constraint("C(Q{},Q{})".format(qi+1,qj+1),[vars[qi], vars[qj]]) 
            sat_tuples = []
            for t in itertools.product(dom, dom):
                if queensCheck(qi, qj, t[0], t[1]):
                    sat_tuples.append(t)
            con.add_satisfying_tuples(sat_tuples)
            cons.append(con)
    
    csp = CSP("{}-Queens".format(n), vars)
    for c in cons:
        csp.add_constraint(c)
    return csp

def test_ord_mrv():

    a = Variable('A', [1])
    b = Variable('B', [1])
    c = Variable('C', [1])
    d = Variable('D', [1])
    e = Variable('E', [1])

    simpleCSP = CSP("Simple", [a,b,c,d,e])

    count = 0
    for i in range(0,len(simpleCSP.vars)):
        simpleCSP.vars[count].add_domain_values(range(0, count))
        count += 1

    var = []
    var = ord_mrv(simpleCSP)

    if var:
        if((var.name) == simpleCSP.vars[0].name):
            print("Passed First Ord MRV Test")
        else:
            print("Failed First Ord MRV Test") 
    else:
        print("No Variable Returned from Ord MRV") 

    a = Variable('A', [1,2,3,4,5])
    b = Variable('B', [1,2,3,4])
    c = Variable('C', [1,2])
    d = Variable('D', [1,2,3])
    e = Variable('E', [1])

    simpleCSP = CSP("Simple", [a,b,c,d,e])

    var = []
    var = ord_mrv(simpleCSP)

    if var:
        if((var.name) == simpleCSP.vars[len(simpleCSP.vars)-1].name):
            print("Passed Second Ord MRV Test")
        else:
            print("Failed Second Ord MRV Test") 
    else:
        print("No Variable Returned from Ord MRV")


def test_model():
    board_1 = [[1,'<',0,'.',0],[0,'.',0,'.',2],[2,'.',0,'>',0]]
    answer_1 = [1,2,3,3,1,2,2,3,1]
    board_2 = [[1,'>',0,'.',3],[0,'.',0,'.',0],[3,'<',0,'.',1]]

    score = 1
    #1st model test
    csp, var_array = futoshiki_csp_model_2(board_1)
    if csp is None:
        print("Failed first model test: wrong solution")
    else:
        solver = BT(csp)
        solver.bt_search(prop_BT)
        sol = []
        for i in range(len(var_array)):
            for j in range(len(var_array)):
                sol.append(var_array[i][j].get_assigned_value())
        if sol == answer_1:
            print("Passed first model test")
        else:
            print("Failed first model test: wrong solution")
    #2nd model test
    csp2, var_array2 = futoshiki_csp_model_2(board_2)
    if csp2 is None:
        print("Failed second model test: wrong solution")
    else:    
        solver = BT(csp2)
        solver.bt_search(prop_BT)
        for i in range(len(var_array2)):
            for j in range(len(var_array2)):
                 if var_array2[i][j].get_assigned_value() is not None:
                    score = 0
        if score == 1:
            print("Passed second model test")
        else:
            print("Failed second model test: 'solved' unsolvable problem")



def queensCheck(qi, qj, i, j):
    '''Return true if i and j can be assigned to the queen in row qi and row qj
       respectively. Used to find satisfying tuples.
    '''
    return i != j and abs(i-j) != abs(qi-qj)

def nQueens(n):
    '''Return an n-queens CSP'''
    i = 0
    dom = []
    for i in range(n):
        dom.append(i+1)

    vars = []
    for i in dom:
        vars.append(Variable('Q{}'.format(i), dom))

    cons = []
    for qi in range(len(dom)):
        for qj in range(qi+1, len(dom)):
            con = Constraint("C(Q{},Q{})".format(qi+1,qj+1),[vars[qi], vars[qj]])
            sat_tuples = []
            for t in itertools.product(dom, dom):
                if queensCheck(qi, qj, t[0], t[1]):
                    sat_tuples.append(t)
            con.add_satisfying_tuples(sat_tuples)
            cons.append(con)

    csp = CSP("{}-Queens".format(n), vars)
    for c in cons:
        csp.add_constraint(c)
    return csp

##Tests FC after the first queen is placed in position 1.
def test_simple_FC():
    did_fail = False
    score = 0
    try:
        queens = nQueens(8)
        curr_vars = queens.get_all_vars()
        curr_vars[0].assign(1)
        prop_FC(queens,newVar=curr_vars[0])
        answer = [[1],[3, 4, 5, 6, 7, 8],[2, 4, 5, 6, 7, 8],[2, 3, 5, 6, 7, 8],[2, 3, 4, 6, 7, 8],[2, 3, 4, 5, 7, 8],[2, 3, 4, 5, 6, 8],[2, 3, 4, 5, 6, 7]]
        var_domain = [x.cur_domain() for x in curr_vars]
        for i in range(len(curr_vars)):
            if var_domain[i] != answer[i]:
                details = "Failed simple FC test: variable domains don't match expected results"
                did_fail = True
                break
        if not did_fail:
            score = 1
            details = ""
    except Exception:
        details = "One or more runtime errors occurred while testing simple FC: %r" % traceback.format_exc()

    return score,details


#@max_grade(1)
##Tests GAC after the first queen is placed in position 1.
def test_simple_GAC():
    did_fail = False
    score = 0
    try:
        queens = nQueens(8)
        curr_vars = queens.get_all_vars()
        curr_vars[0].assign(1)
        prop_GAC(queens,newVar=curr_vars[0])
        answer = [[1],[3, 4, 5, 6, 7, 8],[2, 4, 5, 6, 7, 8],[2, 3, 5, 6, 7, 8],[2, 3, 4, 6, 7, 8],[2, 3, 4, 5, 7, 8],[2, 3, 4, 5, 6, 8],[2, 3, 4, 5, 6, 7]]
        var_domain = [x.cur_domain() for x in curr_vars]
        for i in range(len(curr_vars)):
            if var_domain[i] != answer[i]:
                details = "Failed simple GAC test: variable domains don't match expected results."
                did_fail = True
                break
        if not did_fail:
            score = 1
            details = ""

    except Exception:
        details = "One or more runtime errors occurred while testing simple GAC: %r" % traceback.format_exc()

    return score,details


def three_queen_GAC():
    score = 0
    try:
        queens = nQueens(8)
        curr_vars = queens.get_all_vars()
        curr_vars[0].assign(4)
        curr_vars[2].assign(1)
        curr_vars[7].assign(5)
        prop_GAC(queens)
        answer = [[4],[6, 7, 8],[1],[3, 8],[6, 7],[2, 8],[2, 3, 7, 8],[5]]
        var_vals = [x.cur_domain() for x in curr_vars]

        if var_vals != answer:
            details = "Failed three queens GAC test: variable domains don't match expected results"

        else:
            score = 1
            details = ""
    except Exception:
        details = "One or more runtime errors occurred while testing GAC with three queens: %r" % traceback.format_exc()

    return score,details


def three_queen_FC():
    score = 0
    try:
        queens = nQueens(8)
        curr_vars = queens.get_all_vars()
        curr_vars[0].assign(4)
        curr_vars[2].assign(1)
        curr_vars[7].assign(5)
        prop_FC(queens)

        answer = [[4],[6, 7, 8],[1],[3, 6, 8],[6, 7],[2, 6, 8],[2, 3, 7, 8],[5]]
        var_vals = [x.cur_domain() for x in curr_vars]

        if var_vals != answer:
            details = "Failed three queens FC test: variable domains don't match expected results"

        else:
            score = 1
            details = ""

    except Exception:
        details = "One or more runtime errors occurred while testing FC with three queens: %r" % traceback.format_exc()

    return score,details

if __name__ == "__main__":

 
    #trace = True
    trace = False
    total = 0
    
    print("Testing model")
    test_model()
    print("=======================================================")

    print("Testing ord_mrv heuristic")
    test_ord_mrv()
    print("=======================================================")

    print("FC Test 1: test_simple_FC")
    score,details = test_simple_FC()
    total += score
    print(details)
    print("=======================================================")

    print("GAC Test 1: test_simple_GAC")
    score,details = test_simple_GAC()
    total += score
    print(details)
    print("=======================================================")

    print("FC Test 2: three_queen_FC")
    score,details = three_queen_FC()
    total += score
    print(details)
    print("=======================================================")

    print("GAC Test 2: three_queen_GAC")
    score,details = three_queen_GAC()
    total += score
    print(details)
    print("=======================================================")

    print("Total score on GAC/FC tests: %d/4\n" % total)


