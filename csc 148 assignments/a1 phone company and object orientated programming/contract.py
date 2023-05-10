"""
CSC148, Winter 2022
Assignment 1

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Diane Horton, Jacqueline Smith
"""
import datetime
from math import ceil
from typing import Optional
from bill import Bill
from call import Call


# Constants for the month-to-month contract monthly fee and term deposit
MTM_MONTHLY_FEE = 50.00
TERM_MONTHLY_FEE = 20.00
TERM_DEPOSIT = 300.00

# Constants for the included minutes and SMSs in the term contracts (per month)
TERM_MINS = 100

# Cost per minute and per SMS in the month-to-month contract
MTM_MINS_COST = 0.05

# Cost per minute and per SMS in the term contract
TERM_MINS_COST = 0.1

# Cost per minute and per SMS in the prepaid contract
PREPAID_MINS_COST = 0.025


class Contract:
    """ A contract for a phone line

    This class is not to be changed or instantiated. It is an Abstract Class.

    === Public Attributes ===
    start:
         starting date for the contract
    bill:
         bill for this contract for the last month of call records loaded from
         the input dataset
    contract_type:
         the type of contract
    """
    start: datetime.date
    bill: Optional[Bill]
    contract_type: str

    def __init__(self, start: datetime.date) -> None:
        """ Create a new Contract with the <start> date, starts as inactive
        """
        self.start = start
        self.bill = None
        self.contract_type = ''

    def new_month(self, month: int, year: int, bill: Bill) -> None:
        """ Advance to a new month in the contract, corresponding to <month> and
        <year>. This may be the first month of the contract.
        Store the <bill> argument in this contract and set the appropriate rate
        per minute and fixed cost.

        DO NOT CHANGE THIS METHOD
        """
        raise NotImplementedError

    def bill_call(self, call: Call) -> None:
        """ Add the <call> to the bill.

        Precondition:
        - a bill has already been created for the month+year when the <call>
        was made. In other words, you can safely assume that self.bill has been
        already advanced to the right month+year.
        """
        self.bill.add_billed_minutes(ceil(call.duration / 60.0))

    def cancel_contract(self) -> float:
        """ Return the amount owed in order to close the phone line associated
        with this contract.

        Precondition:
        - a bill has already been created for the month+year when this contract
        is being cancelled. In other words, you can safely assume that self.bill
        exists for the right month+year when the cancelation is requested.
        """
        self.start = None
        return self.bill.get_cost()


class MTMContract(Contract):
    """MTM Contract is a type of contract which only lasts for a month.
    MTM Contract does not have term deposit or free minutes as opposed to the
    Term contract and the prepaid contract.

    === Public attribute ===
    self.start:
        The start date of a month-to-month contract
    self.contract_type:
        The type of a contract
    """
    start: datetime.date
    bill: Optional[Bill]
    contract_type: str

    def __init__(self, start: datetime.date) -> None:
        """create a MTM contract"""
        Contract.__init__(self, start)
        self.contract_type = 'MTM'

    def new_month(self, month: int, year: int, bill: Bill) -> None:
        """To advance a MTM contract to a new month

        preconditions:
        We can assume bill object is created before advancing to a new month
        self.bill.min_rate and self.bill.fixed_cost must be a positive number
        """
        self.bill = bill
        self.bill.type = 'MTM contract'
        self.bill.min_rate = MTM_MINS_COST
        self.bill.fixed_cost = MTM_MONTHLY_FEE


class TermContract(Contract):
    """A term contract is a type of contract that last for a term. Before a term
    starts, a customer must pay certain amount of money as term deposit. The
    contract requires financial commitment until the end of a term contract. If
    the term contract is cancelled ahead of end of term, then term deposited is
    taken away as penalty. Term deposit would be returned if a term contract
    ends after end of a term

    === Public attribute ===
    self.start:
        Start date of the term
    self.term_deposit:
        Amount of fee paid by a customer
    self.end_date_of_contract:
        The end date of a term contract
    self.contract_type:
        The type of contract, which is term contract.
    """
    start: datetime.date
    bill: Optional[Bill]
    contract_type: str
    end_date_of_contract: datetime.date
    term_deposit: float

    def __init__(self, start: datetime.date, end: datetime.date) \
            -> None:
        """Create a Term Contract"""
        Contract.__init__(self, start)
        self.end_date_of_contract = end
        self.contract_type = 'TermContract'

    def new_month(self, month: int, year: int, bill: Bill) -> None:
        """Advance a term contract to a new month

        preconditions:
        We can assume bill object is created before advancing to a new month
        self.bill.free min and self.bill.min_rate and self.bill.fixed_cost are
        positive numbers."""
        self.bill = bill
        self.bill.type = 'TERM'
        self.bill.min_rate = TERM_MINS_COST
        self.bill.fixed_cost += TERM_MONTHLY_FEE
        self.bill.free_min = 0

    def bill_call(self, call: Call) -> None:
        """ Add the <call> to the bill.

        Precondition:
        - a bill has already been created for the month+year when the <call>
        was made. In other words, you can safely assume that self.bill has been
        already advanced to the right month+year.
        """

        if self.bill.free_min + ceil(call.duration / 60.0) <= TERM_MINS:
            self.bill.add_free_minutes(ceil(call.duration / 60.0))
        elif self.bill.free_min == TERM_MINS:
            self.bill.add_billed_minutes(ceil(call.duration / 60.0))
        elif self.bill.free_min + ceil(call.duration / 60) > TERM_MINS and \
                self.bill.free_min < 100:
            gap = TERM_MINS - self.bill.free_min
            self.bill.free_min = TERM_MINS
            gap2 = ceil(call.duration / 60.0) - gap
            self.bill.add_billed_minutes(gap2)

    def cancel_contract(self) -> float:
        """To cancel a term contract. If cancelling date is after the end of the
        contract, term deposit minus the bill of the last month is returned

        preconditions:
        self.end_date_of_contract is a datetime.time object"""
        time = datetime.date(2019, 6, 25)
        self.start = None
        if self.end_date_of_contract <= time:
            return 0.0
        elif self.end_date_of_contract > time:
            return TERM_DEPOSIT - self.bill.get_cost()
        return None


class PrepaidContract(Contract):
    """Prepaid contract is a type of contract. A customer must pay some
    amount of money before start of the contract. The money is then
    reduced by a certain amount in each month until a point that the monthly fee
    is larger than amount of money left. A customer can choose to deposit more
    money any time

    === Public attribute ===
    self.start:
        The start date of a prepaid contract
    self.balance:
        The amount of money stored in a account by a customer.
    self.contract_type:
        The type of a contract.
    """
    start: datetime.date
    bill: Optional[Bill]
    contract_type: str
    balance: float

    def __init__(self, start: datetime.date,
                 balance: float = 0.0) -> None:
        """create a Prepaid contract"""
        Contract.__init__(self, start)
        self.balance = balance
        self.contract_type = 'PrepaidContract'

    def new_month(self, month: int, year: int, bill: Bill) -> None:
        """To advance a prepaid contract to a new month

        preconditions:
        We can assume bill object is created before advancing to a new month.
        """
        self.bill = bill
        self.bill.type = 'MTM'
        if self.balance > -10:
            self.balance = self.balance - 25
        self.bill.min_rate = PREPAID_MINS_COST

    def cancel_contract(self) -> float:
        """To cancel a prepaid contract. If amount owed by a customer is
        positive, then it is returned. If a customer has some money left in a
        account, then the company forfeit it.

        preconditions:
        self.balance is a float
        """
        self.start = None
        if self.balance > 0:
            return self.balance
        else:
            return 0.0


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'allowed-import-modules': [
            'python_ta', 'typing', 'datetime', 'bill', 'call', 'math'
        ],
        'disable': ['R0902', 'R0913'],
        'generated-members': 'pygame.*'
    })
