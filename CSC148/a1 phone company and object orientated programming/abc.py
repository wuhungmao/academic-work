A bill keeps track of the number of calls used each month, along with the corresponding cost per call.
A customer could have multiple phonelines and one id
A call made by a customer to another customer.
a phoneline match to a number and a contract
A phoneline also has a dictionary which contains month+year as keys and bills object as value
A Call has src_number, dst_number, time, duration, src_loc, dst_loc, drawables, connection

This function goes through the loaded data (passed in through the log argument) and instantiates Customer objects, then return them in a list. In the log dictionary, the value corresponding to the customers key contains a list of customers, where each customer is stored as a dictionary itself, with the following keys:

key ‘id’ corresponds to a value representing the customer id
key ‘lines’ corresponds to a list of phone lines for this customer. Each phone line in turn is a list of dictionaries, with the following keys:
    number: the phone number associated with the phone line
contract: the contract type associated with the phone line
