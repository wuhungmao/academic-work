#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete problem solution.  

'''This file will contain different constraint propagators to be used within 
   bt_search.

   propagator == a function with the following template
      propagator(csp, newly_instantiated_variable=None)
           ==> returns (True/False, [(Variable, Value), (Variable, Value) ...]

      csp is a CSP object---the propagator can use this to get access
      to the variables and constraints of the problem. The assigned variables
      can be accessed via methods, the values assigned can also be accessed.

      newly_instanciated_variable is an optional argument.
      if newly_instantiated_variable is not None:
          then newly_instantiated_variable is the most
           recently assigned variable of the search.
      else:
          propagator is called before any assignments are made
          in which case it must decide what processing to do
           prior to any variables being assigned. SEE BELOW

       The propagator returns True/False and a list of (Variable, Value) pairs.
       Return is False if a deadend has been detected by the propagator.
       in this case bt_search will backtrack
       return is true if we can continue.

      The list of variable values pairs are all of the values
      the propagator pruned (using the variable's prune_value method). 
      bt_search NEEDS to know this in order to correctly restore these 
      values when it undoes a variable assignment.

      NOTE propagator SHOULD NOT prune a value that has already been 
      pruned! Nor should it prune a value twice

      PROPAGATOR called with newly_instantiated_variable = None
      PROCESSING REQUIRED:
        for plain backtracking (where we only check fully instantiated 
        constraints) 
        we do nothing...return true, []

        for forward checking (where we only check constraints with one
        remaining variable)
        we look for unary constraints of the csp (constraints whose scope 
        contains only one variable) and we forward_check these constraints.

        for gac we establish initial GAC by initializing the GAC queue
        with all constaints of the csp


      PROPAGATOR called with newly_instantiated_variable = a variable V
      PROCESSING REQUIRED:
         for plain backtracking we check all constraints with V (see csp method
         get_cons_with_var) that are fully assigned.

         for forward checking we forward check all constraints with V
         that have one unassigned variable left

         for gac we initialize the GAC queue with all constraints containing V.
		 
		 
var_ordering == a function with the following template
    var_ordering(csp)
        ==> returns Variable 

    csp is a CSP object---the heuristic can use this to get access to the
    variables and constraints of the problem. The assigned variables can be
    accessed via methods, the values assigned can also be accessed.

    var_ordering returns the next Variable to be assigned, as per the definition
    of the heuristic it implements.
   '''

def prop_BT(csp, newVar=None):
    '''Do plain backtracking propagation. That is, do no 
    propagation at all. Just check fully instantiated constraints'''
    
    if not newVar:
        return True, []
    for c in csp.get_cons_with_var(newVar):
        if c.get_n_unasgn() == 0:
            vals = []
            vars = c.get_scope()
            for var in vars:
                vals.append(var.get_assigned_value())
            if not c.check(vals):
                return False, []
    return True, []

def FCcheck(constraint, var, prune_bookkeeping):
    for val in var.cur_domain():
        var.assign(val)
        sat_value = []
        for all_var in constraint.get_scope():
            sat_value.append(all_var.get_assigned_value())
        if constraint.check(sat_value) == False:
            var.prune_value(val)
            prune_bookkeeping.append([var, val])
        var.unassign()
    if var.cur_domain() == []:
        return True
    return False
            
            
# // C is a constraint with all its variables already
# // assigned, except for variable X.
# 1. for d := each member of CurDom(X):
# 2. if making X = d together with previous assignments
# to variables in the scope of C falsifies C:
# 3. remove d from CurDom(X)
# 4. if CurDom[X] == {}:
# 5. RETURN DWO # Domain Wipe Out
# 6. RETURN ok

def prop_FC(csp, newVar=None):
    '''Do forward checking. That is check constraints with 
       only one uninstantiated variable. Remember to keep 
       track of all pruned variable,value pairs and return '''
    #IMPLEMENT
    if newVar is not None:
    #newVar is not None, which means we find all constraints with only one uninstantiated variable and has
    #newVar in its scope.
        constraint_lst = []
        for constraint in csp.get_cons_with_var(newVar):
            if constraint.get_n_unasgn() == 1:
                constraint_lst.append(constraint)     
        DWOoccurred = False
        prune_bookkeeping = []
        for constraint in constraint_lst:
            the_unassigned_var = constraint.get_unasgn_vars()[0]
            if FCcheck(constraint, the_unassigned_var, prune_bookkeeping) == True:
                DWOoccurred = True
                break
        if DWOoccurred == True:
            #domain wipe out, unassign variables and restore all values pruned
            return (False, prune_bookkeeping)
        else:
            return (True, prune_bookkeeping)
    else:
    #newVar is None, which means we find all constraints with only one uninstantiated variable
        constraint_lst = []
        for var in csp.get_all_unasgn_vars():
            for constraint in csp.get_cons_with_var(var):
                num_unassigned_vars = constraint.get_n_unasgn()
                if constraint.get_n_unasgn() == 1:
                    constraint_lst.append(constraint)
        DWOoccurred = False
        prune_bookkeeping = []
        lst_var = csp.get_all_vars()
        for constraint in constraint_lst:
            lst_unassigned_vars = constraint.get_unasgn_vars()
            the_unassigned_var = constraint.get_unasgn_vars()[0]
            if FCcheck(constraint, the_unassigned_var, prune_bookkeeping) == True:
                DWOoccurred = True
                break
        if DWOoccurred == True:
            #domain wipe out, unassign variables and restore all values pruned
            return (False, prune_bookkeeping)
        else:
            return (True, prune_bookkeeping)
            
    


    # V := PickUnassignedVariable()
    # Assigned[V] := TRUE
    # for d := each member of CurDom(V)
    # Value[V] := d
    # DWOoccured:= False
    # for each constraint C over V such that C has only one
    # unassigned variable X in its scope:
    # if FCCheck(C,X) == DWO: # X domain becomes empty
    # DWOoccurred:= True
    # BREAK # stop checking constraints
    # if NOT DWOoccured: # all constraints were ok
    # FC(Level+1)
    # RestoreAllValuesPrunedByFCCheck()
    # Assigned[V] := FALSE # UNDO as we have tried all of V’s values
    # RETURN
    

def prop_GAC(csp, newVar=None):
    '''Do GAC propagation. If newVar is None we do initial GAC enforce 
       processing all constraints. Otherwise we do GAC enforce with
       constraints containing newVar on GAC Queue'''
       
    if newVar is not None:
        GACQueue = []
        for constraint in csp.get_cons_with_var(newVar):
            GACQueue.append(constraint)
    else:
        GACQueue = []        
        GACQueue.extend(csp.get_all_cons())

    prune_bookkeeping = []
    while len(GACQueue) != 0:
        constraint = GACQueue.pop(0)
        for var in constraint.get_unasgn_vars():
            for val in var.cur_domain():
                if constraint.has_support(var, val) == False:
                    var.prune_value(val)
                    prune_bookkeeping.append([var, val])
                    if var.cur_domain() == []:
                        return (False, prune_bookkeeping)
                    else:
                        for new_constraint in csp.get_cons_with_var(var):
                            if new_constraint not in GACQueue:
                                GACQueue.append(new_constraint)
    return (True, prune_bookkeeping)

def ord_mrv(csp):
    ''' return variable according to the Minimum Remaining Values heuristic '''
    lst_var = csp.get_all_vars()
    dom_size = lst_var[0].cur_domain_size()
    the_var = lst_var[0]
    for var in lst_var:
        if dom_size > var.cur_domain_size():
            dom_size = var.cur_domain_size()
            the_var = var
    return the_var

	