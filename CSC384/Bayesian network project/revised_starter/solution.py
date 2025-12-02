import random #for sampling methods
import pandas as pd #to simplify reading and holding data
from bnetbase import Variable, Factor, BN, restrict_factor, sum_out_variable, normalize

def multiply_factors(Factors):
    '''return a new factor that is the product of the factors in Fators'''
    #YOUR CODE HERE
    pass #replace this!

###Orderings
def min_fill_ordering(Factors, QueryVar):
    '''Compute a min fill ordering given a list of factors. Return a list
    of variables from the scopes of the factors in Factors. The QueryVar is
    NOT part of the returned ordering'''
    #YOUR CODE HERE
    pass #replace this!

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
   mean that Pr(A='a'|B=1, C='c') = 0.5 Pr(A='a'|B=1, C='c') = 0.24
   Pr(A='a'|B=1, C='c') = 0.26

    '''
    #YOUR CODE HERE
    pass #replace this!

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

    # YOUR CODE HERE
    pass #replace this!

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

    #YOUR CODE HERE
    pass #replace this!


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

    #YOUR CODE HERE
    pass #replace this!


if __name__ == "__main__":

    #You can Calculate Causal Effects of Country on Fatality here!
    
    pass #replace this!



