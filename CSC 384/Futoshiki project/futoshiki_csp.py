#Look for #IMPLEMENT tags in this file.
'''
All models need to return a CSP object, and a list of lists of Variable objects 
representing the board. The returned list of lists is used to access the 
solution. 

For example, after these three lines of code
 
    csp, var_array = futoshiki_csp_model_1(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the Futoshiki puzzle.

1. futoshiki_csp_model_1 (worth 20/100 marks)
    - A model of a Futoshiki grid built using only 
      binary not-equal constraints for both the row and column constraints.

2. futoshiki_csp_model_2 (worth 20/100 marks)
    - A model of a Futoshiki grid built using only n-ary 
      all-different constraints for both the row and column constraints. 

'''
from cspbase import *
import itertools

def model1_check(input1, input2, cond):
    if cond == '>':
        return input1 > input2
    elif cond == '<':
        return input1 < input2
    elif cond == '!=':
        return input1 != input2

def futoshiki_csp_model_1(futo_grid):
    # board_1 = [[1,'<',0,'.',0],[0,'.',0,'.',2],[2,'.',0,'>',0]]
    # answer_1 = [1,2,3,3,1,2,2,3,1]
    # board_2 = [[1,'>',0,'.',3],[0,'.',0,'.',0],[3,'<',0,'.',1]]
    
    n = len(futo_grid)
    offset = 2
    dom = list(range(1, n+1))
    var_lst = []
    #initialize variables
    for i in range(len(futo_grid)):
        for j in range(len(futo_grid[0])):
            if isinstance(futo_grid[i][j], int):
                if futo_grid[i][j] == 0:
                    var_lst.append(Variable('Q{}'.format(i*n + j//offset + 1), dom))
                else:
                    new_var = Variable('Q{}'.format(i*n + j//offset + 1), [futo_grid[i][j]])
                    new_var.assign(futo_grid[i][j])
                    var_lst.append(new_var)
    
    csp = CSP("model_1", var_lst)
    
    sec_var_lst = []
    #initialize row constraints
    constr_lst = []
    for k in range(n):
        var_row = var_lst[k*n:(k+1)*n]
        row_perm = itertools.combinations(var_row, 2)
        for scope in row_perm:
            constraint = Constraint("C({},{})".format(scope[0].name,scope[1].name), scope)
            sat_tuples = []
            for prod in itertools.product(dom, dom):
                if model1_check(prod[0], prod[1], '!='):
                    sat_tuples.append(prod)
            constraint.add_satisfying_tuples(sat_tuples)
            constr_lst.append(constraint)
        sec_var_lst.append(var_row)
    
    #initialize column constraints
    for l in range(n):
        var_col = var_lst[l:len(var_lst):n]
        col_perm = itertools.combinations(var_col, 2)
        for scope in col_perm:
            constraint = Constraint("C({},{})".format(scope[0].name,scope[1].name), scope)
            sat_tuples = []
            for prod in itertools.product(dom, dom):
                if model1_check(prod[0], prod[1], '!='):
                    sat_tuples.append(prod)
            constraint.add_satisfying_tuples(sat_tuples)
            constr_lst.append(constraint)
            
    #initialize special constraints
    curr_ind = 0
    for i in range(len(futo_grid)):
        for j in range(len(futo_grid[0])):
            if isinstance(futo_grid[i][j], int):
                curr_ind+=1
            elif isinstance(futo_grid[i][j], str):
                if futo_grid[i][j] == '>':
                    var1 = var_lst[curr_ind-1]
                    var2 = var_lst[curr_ind]
                    spec_constraint1 = Constraint("C({},{})".format(var1.name,var2.name), [var1,var2])
                    sat_tuples = []
                    for prod in itertools.product(dom, dom):
                        if model1_check(prod[0], prod[1], '>'):
                            sat_tuples.append(prod)
                    spec_constraint1.add_satisfying_tuples(sat_tuples)
                    constr_lst.append(spec_constraint1)
                elif futo_grid[i][j] == '<':
                    var1 = var_lst[curr_ind-1]
                    var2 = var_lst[curr_ind]
                    spec_constraint2 = Constraint("C({},{})".format(var1.name,var2.name), [var1,var2])
                    sat_tuples = []
                    for prod in itertools.product(dom, dom):
                        if model1_check(prod[0], prod[1], '<'):
                            sat_tuples.append(prod)
                    spec_constraint2.add_satisfying_tuples(sat_tuples)
                    constr_lst.append(spec_constraint2)
                    
    #add all constraints into the CSP
    for i in range(len(constr_lst)):
        csp.add_constraint(constr_lst[i])
    
    return csp, sec_var_lst 
    
    
def model2_check(lst, cond):
    if cond == '>':
        return lst[0] > lst[1]
    elif cond == '<':
        return lst[0] < lst[1]
    elif cond == '!=':
        for i in range(len(lst)-1):
            if lst[i] == lst[i+1]:
                return False
        return True

def futoshiki_csp_model_2(futo_grid):
    # board_1 = [[1,'<',0,'.',0],[0,'.',0,'.',2],[2,'.',0,'>',0]]
    # answer_1 = [1,2,3,3,1,2,2,3,1]
    # board_2 = [[1,'>',0,'.',3],[0,'.',0,'.',0],[3,'<',0,'.',1]]
    
    n = len(futo_grid)
    offset = 2
    dom = list(range(1, n+1))
    var_lst = []
    #initialize variables
    for i in range(len(futo_grid)):
        for j in range(len(futo_grid[0])):
            if isinstance(futo_grid[i][j], int):
                if futo_grid[i][j] == 0:
                    var_lst.append(Variable('Q{}'.format(i*n + j//offset + 1), dom))
                else:
                    new_var = Variable('Q{}'.format(i*n + j//offset + 1), [futo_grid[i][j]])
                    new_var.assign(futo_grid[i][j])
                    var_lst.append(new_var)
    
    csp = CSP("model_2", var_lst)
    
    sec_var_lst = []
    #initialize row constraints
    constr_lst = []
    for k in range(n):
        var_row = var_lst[k*n:(k+1)*n]
        name = "C("
        for a in range(len(var_row)):
             name += var_row[a].name + ","
        name = name[:-1] + ")"
        
        constraint = Constraint(name, var_row)
        sat_tuples = []
        for prod in itertools.permutations(dom, n):
            if model2_check(prod, '!='):
                sat_tuples.append(prod)
        constraint.add_satisfying_tuples(sat_tuples)
        constr_lst.append(constraint)
        
        sec_var_lst.append(var_row)
    
    #initialize column constraints
    for l in range(n):
        var_col = var_lst[l:len(var_lst):n]
        
        name = "C("
        for a in range(len(var_col)):
             name += var_col[a].name + ","
        name = name[:-1] + ")"
        
        constraint = Constraint(name, var_col)
        sat_tuples = []
        for prod in itertools.permutations(dom, n):
            if model2_check(prod, '!='):
                sat_tuples.append(prod)
        constraint.add_satisfying_tuples(sat_tuples)
        constr_lst.append(constraint)
        
    #initialize special constraints
    curr_ind = 0
    for i in range(len(futo_grid)):
        for j in range(len(futo_grid[0])):
            if isinstance(futo_grid[i][j], int):
                curr_ind+=1
            elif isinstance(futo_grid[i][j], str):
                if futo_grid[i][j] == '>':
                    var1 = var_lst[curr_ind-1]
                    var2 = var_lst[curr_ind]
                    spec_constraint1 = Constraint("C({},{})".format(var1.name,var2.name), [var1,var2])
                    sat_tuples = []
                    for prod in itertools.product(dom, dom):
                        if model2_check([prod[0], prod[1]], '>'):
                            sat_tuples.append(prod)
                    spec_constraint1.add_satisfying_tuples(sat_tuples)
                    constr_lst.append(spec_constraint1)
                elif futo_grid[i][j] == '<':
                    var1 = var_lst[curr_ind-1]
                    var2 = var_lst[curr_ind]
                    spec_constraint2 = Constraint("C({},{})".format(var1.name,var2.name), [var1,var2])
                    sat_tuples = []
                    for prod in itertools.product(dom, dom):
                        if model2_check([prod[0], prod[1]], '<'):
                            sat_tuples.append(prod)
                    spec_constraint2.add_satisfying_tuples(sat_tuples)
                    constr_lst.append(spec_constraint2)
                    
    #add all constraints into the CSP
    for i in range(len(constr_lst)):
        csp.add_constraint(constr_lst[i])
    
    return csp, sec_var_lst    
