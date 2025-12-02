import itertools
import random #for sampling methods
import pandas as pd #to simplify reading and holding data
from bnetbase import Variable, Factor, BN, restrict_factor, sum_out_variable, normalize

def cartesian_product(*domains):
    return list(itertools.product(*domains))

def multiply_factors(Factors, evidence_variable = []):
    '''return a new factor that is the product of the factors in Factors'''
    # Find the scope of multiplied factor
    new_scope = []
    for factor in Factors:
        for var in factor.get_scope():
            if var not in new_scope:
                new_scope.append(var)
    
    # create a new name for multiplied factor
    new_name = ""
    for factor in Factors:
        new_name += factor.name
    
    new_factor = Factor(new_name, new_scope)
    
    domain_lst = []
    
    for var in new_scope:
        if var not in evidence_variable:
            domain_lst.append(var.domain())
        else:
            domain_lst.append([var.get_evidence()])
    
    # create all possible assignments to all variables inside multiplied factor
    possible_assignments = cartesian_product(*domain_lst)
    
    # Calculate probabilities using factors and variables assignment and add corresponding value
    # to multiplied factor 
    for assignment in possible_assignments:
        for i, assignment_val in enumerate(assignment):
            new_scope[i].set_assignment(assignment_val)
        sum = 1
        for factor in Factors:
            sum*=factor.get_value_at_current_assignments()
        new_factor.add_value_at_current_assignment(sum)
        
    return new_factor
    
###Orderings
def min_fill_ordering(Factors, QueryVar, evidence_variable):
    '''Compute a min fill ordering given a list of factors. Return a list
    of variables from the scopes of the factors in Factors. The QueryVar is
    NOT part of the returned ordering'''
    All_var_to_be_eliminated = []
    for ind_factor in Factors:
        for var in ind_factor.get_scope():
            if var not in All_var_to_be_eliminated and var != QueryVar and var not in evidence_variable:
                All_var_to_be_eliminated.append(var)
    num_var = len(All_var_to_be_eliminated)

    var_lst = []
    for _ in range(num_var):
        # eliminate every variable in scope, each time find the variable that creates
        # smallest factor upon elimnation
        smallest_count = 1000
        all_factors_that_include_var_to_be_eliminated = []
        
        # find the variable that creates smallest factor upon elimination
        # Also find the factors that include the variable

        # find the variable that create smallest factor
        for var_to_be_eliminated in All_var_to_be_eliminated:
            all_factors_that_include_var_to_be_eliminated_temp = set()
            all_var_in_factor_if_remov_var = set()
            for factor in Factors:
                if var_to_be_eliminated in factor.get_scope():
                    all_factors_that_include_var_to_be_eliminated_temp.add(factor)
                    for candidate_var in factor.get_scope():
                        all_var_in_factor_if_remov_var.add(candidate_var)
                        
            if smallest_count > len(all_var_in_factor_if_remov_var):
                smallest_count = len(all_var_in_factor_if_remov_var)
                var_to_be_eliminated_that_create_smallest_factor = var_to_be_eliminated
                all_factors_that_include_var_to_be_eliminated = list(all_factors_that_include_var_to_be_eliminated_temp)
                
        # Add the variable into var_lst and create new factor based on all factors that 
        # include the variable
        var_lst.append(var_to_be_eliminated_that_create_smallest_factor)
        All_var_to_be_eliminated.remove(var_to_be_eliminated_that_create_smallest_factor) 
        for factor_to_be_remove in all_factors_that_include_var_to_be_eliminated:
            Factors.remove(factor_to_be_remove)
        new_factor_before_sum_out = multiply_factors(all_factors_that_include_var_to_be_eliminated, evidence_variable)
        new_factor_after_sum_out = sum_out_variable(new_factor_before_sum_out, var_to_be_eliminated_that_create_smallest_factor)
        Factors.append(new_factor_after_sum_out)
        
    return var_lst        

def VE(Net, QueryVar, EvidenceVars):
    '''
    Input: Net---a BN object (a Bayes Net)
           QueryVar---a Variable object (the variable whose distribution
                      we want to compute)
           EvidenceVars---a LIST of Variable objects. Each of these
                          variables has had its evidence set to a particular
                          value from its domain using set_evidence.

    VE returns a distribution over the values of QueryVar, i.e., a list
    of numbers one for every value in QueryVar's domain. These numbers
    sum to one, and the i'th number is the probability that QueryVar is
    equal to its i'th value given the setting of the evidence
    variables. For example if QueryVar = A with Dom[A] = ['a', 'b',
    'c'], EvidenceVars = [B, C], and we have previously called
    B.set_evidence(1) and C.set_evidence('c'), then VE would return a
    list of three numbers. E.g. [0.5, 0.24, 0.26]. These numbers would
    mean that Pr(A='a'|B=1, C='c') = 0.5 Pr(A='b'|B=1, C='c') = 0.24
    Pr(A='c'|B=1, C='c') = 0.26
    '''     
    # Restrict factor so that evidence variables are restricted to a specific value
    net_factors1 = Net.factors()
    for factor in net_factors1:
        for evidence_var in EvidenceVars:
            if evidence_var in factor.get_scope():
                restrict_factor(factor, evidence_var, evidence_var.get_evidence())
    
    net_factors2 = list(net_factors1)
    # Find the order which creates the smallest number of factors at each step
    lst_of_var = min_fill_ordering(net_factors1, QueryVar, EvidenceVars)
    
    # Start doing variable elimination based on the list of variable we got from min_fill_ordering
    for smal_factor_var in lst_of_var:
        all_factors_inclu_z = []
        net_factors2_cpy = list(net_factors2)
        for factor in net_factors2_cpy:
            if smal_factor_var in factor.get_scope():
                all_factors_inclu_z.append(factor)
                net_factors2.remove(factor)
        new_factor1 = multiply_factors(all_factors_inclu_z, EvidenceVars)
        new_factor2 = sum_out_variable(new_factor1, smal_factor_var)
        net_factors2.append(new_factor2)
    final_lst = []
    
    # at this point net_factors2 would only contain factors that 
    # include query variable and evidence variables
    # Starting adding probability based on the value in the domain of query variable
    for evidence_variable in EvidenceVars:
        evidence_variable.set_assignment(evidence_variable.get_evidence())

    for domain_val in QueryVar.domain():
        QueryVar.set_assignment(domain_val)
        sum = 1
        for left_factor in net_factors2:
            sum *= left_factor.get_value_at_current_assignments()
        final_lst.append(sum)
    return normalize(final_lst)
    
# This helper function basically compute w and randomly select a value 
# for each variable except evidence variable
def helper_functions(Net, EvidenceVars):
    w = 1
    select_pool_var = Net.variables()
    select_pool_fac = Net.factors()
    selected_var = []
    while len(select_pool_var) != 0:
        selected_var = select_pool_var.pop(0)
        this_factor = True
        for factor_in_pool in select_pool_fac:
            if selected_var in factor_in_pool.get_scope():
                if len(factor_in_pool.get_scope()) == 1:
                    selected_factor = factor_in_pool
                    this_factor = True
                else:
                    for all_other_var in factor_in_pool.get_scope():
                        if all_other_var == selected_var:
                            continue
                        else:
                            if all_other_var.get_assignment() == None:
                                this_factor = False
                                break
            else:
                continue
            if this_factor == True:
                selected_factor = factor_in_pool
                break
                    
        lst_probabilities = []
        for domain_val in selected_var.domain():
            selected_var.set_assignment(domain_val)
            lst_probabilities.append(selected_factor.get_value_at_current_assignments())
                                     
        # Choose a value out of domain of a variable based on probability distribution
        # over domain values
        selected_val = random.choices(selected_var.domain(), lst_probabilities)[0]
        
        if selected_var in EvidenceVars:
            selected_var.set_assignment(selected_var.get_evidence())
            w *= selected_factor.get_value_at_current_assignments()
        else:
            selected_var.set_assignment(selected_val)
    return w

def SampleBN(Net, QueryVar, EvidenceVars):
    '''
     Input: Net---a BN object (a Bayes Net)
            QueryVar---a Variable object (the variable whose distribution
                       we want to compute)
            EvidenceVars---a LIST of Variable objects. Each of these
                           variables has had its evidence set to a particular
                           value from its domain using set_evidence.

    SampleBN returns a distribution over the values of QueryVar, i.e., a list
    of numbers one for every value in QueryVar's domain. These numbers
    sum to one, and the i'th number is the probability that QueryVar is
    equal to its i'th value given the setting of the evidence
    variables.

    SampleBN should generate **1000** samples using the likelihood
    weighting method described in class.  It should then calculate
    a distribution over value assignments to QueryVar based on the
    values of these samples.
    '''
    
    sum = 0
    query_var_domain = dict()
    for dom_val1 in QueryVar.domain():
        query_var_domain[dom_val1] = 0
        
    # run the sampling 1000 times. 
    # Each time query variable get a different value assigned
    # sum is used to calculate total w
    for _ in range(1000):
        w = helper_functions(Net, EvidenceVars)
        sum+=w
        query_var_domain[QueryVar.get_assignment()]+=w
    
    # calculate the probability of each domain value and record them in a list and return
    final_lst = []
    for dom_val2 in QueryVar.domain():
        final_lst.append(query_var_domain[dom_val2]/sum)
    
    return final_lst
    
def CausalModelConfounder():
    """CausalModelConfounder returns a Causal model (i.e. a BN) that
   represents the joint distribution of value assignments to
   variables in COVID-19 data.

   The structure of this model should reflect the assumption
   that age is a COUNFOUNDING variable in the network, and
   is therefore a cause of both Country and Fatality.

    Returns:
        BN: A BN that represents the causal model
    """

    ### READ IN THE DATA
    df = pd.read_csv('data/covid.csv')

    ### DOMAIN INFORMATION
    variable_domains = {
    "Age": ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80'],
    "Country": ['Italy', 'China'],
    "Fatality": ['YES', 'NO']
    }
    
    # initialize three variables
    age_var = Variable("Age", variable_domains['Age'])
    coun_var = Variable("Country", variable_domains['Country'])
    fatal_var = Variable("Fatality", variable_domains['Fatality'])
    
    # initialize three factors 
    age_fac = Factor("P(A)", [age_var])
    coun_age_fac = Factor("P(C|A)", [coun_var, age_var])
    coun_age_fatal_fac = Factor("P(F|C, A)", [coun_var, age_var, fatal_var])
    
    # create the bayes net
    Net = BN("age as confounder", [coun_var, age_var, fatal_var], [age_fac, coun_age_fac, coun_age_fatal_fac])
    
    # Calculate the probability of each domain value of age variable
    age_dict = {}
    for age_range1 in variable_domains['Age']:
        age_dict[age_range1] = 0
    sum = 0
    for _, row in df.iterrows():
        age_var = row['Age']
        age_dict[age_var]+=1
        sum += 1
    final_lst = []
    for age_range2 in variable_domains['Age']:
        final_lst.append([age_range2, age_dict[age_range2]/sum])
        
    age_fac.add_values(final_lst)
    
    # Calculate the probability of any country given any age
    # coun_age_dict contains the number of each pair of country and age, 
    # age_dict contains the number of each age range
    coun_age_dict = {}
    for country1 in variable_domains['Country']:
        for age_range3 in variable_domains['Age']:
            coun_age_dict[(country1, age_range3)] = 0
    for _, row in df.iterrows():
        coun_age_dict[(row['Country'], row['Age'])]+=1
    final_lst2 = []
    for age_range4 in variable_domains['Age']:
        final_lst2.append(['Italy', age_range4, coun_age_dict[('Italy', age_range4)]/age_dict[age_range4]])
        final_lst2.append(['China', age_range4, coun_age_dict[('China', age_range4)]/age_dict[age_range4]])
    coun_age_fac.add_values(final_lst2)
    
    # Calculate the probability of any country given any age
    # Similarly, count_age_fatal_dict contains the number of any country, age, fatality pair
    # This uses dynamic programming to reduce the runtime. In specific, coun_age_dict 
    # is used here again to compute the probability of fatality given any country and age
    count_age_fatal_dict = {}
    for country2 in variable_domains['Country']:
        for age_range5 in variable_domains['Age']:
            for fatality in variable_domains['Fatality']:
                count_age_fatal_dict[(country2, age_range5, fatality)] = 0 
    
    for _, row in df.iterrows():
        count_age_fatal_dict[(row['Country'], row['Age'], row['Fatality'])]+=1
    
    final_lst3 = []
    for country3 in variable_domains['Country']:
        for age_range6 in variable_domains['Age']:
            final_lst3.append([country3, age_range6, 'YES', count_age_fatal_dict[(country3, age_range6, 'YES')]/coun_age_dict[country3, age_range6]])
            final_lst3.append([country3, age_range6, 'NO', count_age_fatal_dict[(country3, age_range6, 'NO')]/coun_age_dict[country3, age_range6]])
    coun_age_fatal_fac.add_values(final_lst3)
    
    return Net


def CausalModelMediator():
    """CausalModelConfounder returns a Causal model (i.e. a BN) that
    represents the joint distribution of value assignments to
    variables in COVID-19 data.

    The structure of this model should reflect the assumption
    that age is a MEDIATING variable in the network, and
    is mediates the causal effect of Country on Fatality.

     Returns:
         BN: A BN that represents the causal model
     """

    ### READ IN THE DATA
    df = pd.read_csv('data/covid.csv')

    ### DOMAIN INFORMATION
    variable_domains = {
    "Age": ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80'],
    "Country": ['Italy', 'China'],
    "Fatality": ['YES', 'NO']
    }
    
    age_var = Variable("Age", variable_domains['Age'])
    coun_var = Variable("Country", variable_domains['Country'])
    fatal_var = Variable("Fatality", variable_domains['Fatality'])
    
    coun_fac = Factor("P(C)", [coun_var])
    age_coun_fac = Factor("P(A|C)", [age_var, coun_var])
    coun_age_fatal_fac = Factor("P(F|A, C)", [coun_var, age_var, fatal_var])
    
    Net = BN("age as mediator", [coun_var, age_var, fatal_var], [coun_fac, age_coun_fac, coun_age_fatal_fac])
    
    #calculate probability distribution
    coun_dict = dict()
    num_of_italy = (df["Country"] == 'Italy').sum()
    num_of_china = (df["Country"] == 'China').sum()
    coun_dict['Italy'] = num_of_italy
    coun_dict['China'] = num_of_china
    coun_fac.add_values([["Italy", num_of_italy/(num_of_italy+num_of_china)], ["China", num_of_china/(num_of_italy+num_of_china)]])
    
    #calculate probability distribution
    age_coun_dict = dict()
    for age_range in variable_domains['Age']:
        for country in variable_domains['Country']:
            age_coun_dict[age_range, country] = 0
            
    for _, row in df.iterrows():
        age_coun_dict[(row['Age'], row['Country'])]+=1

    final_lst = []
    for age_range in variable_domains['Age']:
        for country in variable_domains['Country']:
            final_lst.append([age_range, country, age_coun_dict[(age_range, country)]/coun_dict[country]])
    age_coun_fac.add_values(final_lst)
    
    #calculate probability distribution
    coun_age_fatal_dict = {}
    for fatal in variable_domains['Fatality']:
        for age_range in variable_domains['Age']:
            for country in variable_domains['Country']:
                coun_age_fatal_dict[(country, age_range, fatal)] = 0
            
    for _, row in df.iterrows():
        coun_age_fatal_dict[(row['Country'], row['Age'], row['Fatality'])]+=1
    
    final_lst2 = []
    for fatal in variable_domains['Fatality']:
        for age_range in variable_domains['Age']:
            for country in variable_domains['Country']:
                final_lst2.append([country, age_range, fatal, coun_age_fatal_dict[(country, age_range, fatal)]/age_coun_dict[(age_range, country)]])
    coun_age_fatal_fac.add_values(final_lst2)
    
    return Net

if __name__ == "__main__":
    # running this may take a while

    # the effect of “Country” is mediated by age
    model = CausalModelMediator()
    
    Country = model.variables()[0]
    Fatality = model.variables()[2]

    Country.set_evidence('Italy')
    #This exact value is the probability of fatality being Yes and country being italy
    exact1 = VE(model, Fatality, [Country])[0]
    
    Country.set_evidence('China')
    #This exact value is the probability of fatality being Yes and country being china
    exact2 = VE(model, Fatality, [Country])[0]
    
    ACE = exact1 - exact2
    
    Country.set_evidence('Italy')
    estimated1 = SampleBN(model, Fatality, [Country])[0]
    Country.set_evidence('China')
    estimated2 = SampleBN(model, Fatality, [Country])[0]
    
    ACE2 = estimated1 - estimated2
    
    print("Mediator: \nExact value is " + str(ACE) + "\nEstimated value is " + str(ACE2))

# ----------------------------------------------------------------
    # the effect of “Country” is confounded by age

    model = CausalModelConfounder()

    Country = model.variables()[0]
    Age = model.variables()[1]
    Fatality = model.variables()[2]

    sum = 0
    P_Age_exact_value = VE(model, Age, [])
    for i, age_domain_val in enumerate(Age.domain()):
        Age.set_evidence(age_domain_val)
        Country.set_evidence('Italy')
        P_F_YES__Country_Italy_Age_exact_value = VE(model, Fatality, [Country, Age])[0]
        Country.set_evidence('China')
        P_F_YES__Country_China_Age_exact_value  = VE(model, Fatality, [Country, Age])[0]
        sum += P_Age_exact_value[i]*(P_F_YES__Country_Italy_Age_exact_value - P_F_YES__Country_China_Age_exact_value)
    
    print("Confounder: \nExact value is " + str(sum))
    
    sum = 0
    P_Age_exact_value = SampleBN(model, Age, [])
    for i, age_domain_val in enumerate(Age.domain()):
        Age.set_evidence(age_domain_val)
        Country.set_evidence('Italy')
        P_F_YES__Country_Italy_Age_exact_value = SampleBN(model, Fatality, [Country, Age])[0]
        Country.set_evidence('China')
        P_F_YES__Country_China_Age_exact_value  = SampleBN(model, Fatality, [Country, Age])[0]
        sum += P_Age_exact_value[i]*(P_F_YES__Country_Italy_Age_exact_value - P_F_YES__Country_China_Age_exact_value)
    
    print("Estimated value is " + str(sum))
    




