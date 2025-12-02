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
import time
import datetime
from call import Call
from customer import Customer


class Filter:
    """ A class for filtering customer data on some criterion. A filter is
    applied to a set of calls.

    This is an abstract class. Only subclasses should be instantiated.
    """
    def __init__(self) -> None:
        pass

    def apply(self, customers: list[Customer],
              data: list[Call],
              filter_string: str) \
            -> list[Call]:
        """ Return a list of all calls from <data>, which match the filter
        specified in <filter_string>.

        The <filter_string> is provided by the user through the visual prompt,
        after selecting this filter.
        The <customers> is a list of all customers from the input dataset.

         If the filter has
        no effect or the <filter_string> is invalid then return the same calls
        from the <data> input.

        Precondition:
        - <customers> contains the list of all customers from the input dataset
        - all calls included in <data> are valid calls from the input dataset
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        raise NotImplementedError


class ResetFilter(Filter):
    """
    A class for resetting all previously applied filters, if any.
    """
    def apply(self, customers: list[Customer], data: list[Call],
              filter_string: str) -> list[Call]:
        """ Reset all of the applied filters. Return a List containing all the
        calls corresponding to <customers>.
        The <data> and <filter_string> arguments for this type of filter are
        ignored.

        Precondition:
        - <customers> contains the list of all customers from the input dataset
        """
        filtered_calls = []
        for c in customers:
            customer_history = c.get_history()
            # only take outgoing calls, we don't want to include calls twice
            filtered_calls.extend(customer_history[0])
        return filtered_calls

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        return "Reset all of the filters applied so far, if any"


class CustomerFilter(Filter):
    """
    A class for selecting only the calls from a given customer.
    """
    def apply(self, customers: list[Customer],
              data: list[Call],
              filter_string: str) \
            -> list[Call]:
        """ Return a list of all unique calls from <data> made or
        received by the customer with the id specified in <filter_string>.

        The <customers> list contains all customers from the input dataset.

        The filter string is valid if and only if it contains a valid
        customer ID.
        - If the filter string is invalid, return the original list <data>
        - If the filter string is invalid, your code must not crash, as
        specified in the handout.

        Do not mutate any of the function arguments!
        """
        returned_data = []
        customer_with_correct_id = ''
        if filter_string == '':
            return data
        try:
            int(filter_string)
            len(filter_string)
        except ValueError:
            return data
        except TypeError:
            return data
        else:
            if len(filter_string) != 4:
                return data
            for customer in customers:
                if customer.get_id() == int(filter_string):
                    customer_with_correct_id = customer
            if customer_with_correct_id == '':
                return data
            number_list = customer_with_correct_id.get_phone_numbers()
            for call in data:
                if call.dst_number in number_list \
                        or call.src_number in number_list:
                    returned_data.append(call)
            return returned_data

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        return "Filter events based on customer ID"


class DurationFilter(Filter):
    """
    A class for selecting only the calls lasting either over or under a
    specified duration.
    """
    def apply(self, customers: list[Customer],
              data: list[Call],
              filter_string: str) \
            -> list[Call]:
        """ Return a list of all unique calls from <data> with a duration
        of under or over the time indicated in the <filter_string>.

        The <customers> list contains all customers from the input dataset.

        The filter string is valid if and only if it contains the following
        input format: either "Lxxx" or "Gxxx", indicating to filter calls less
        than xxx or greater than xxx seconds, respectively.
        - If the filter string is invalid, return the original list <data>
        - If the filter string is invalid, your code must not crash, as
        specified in the handout.

        Do not mutate any of the function arguments!
        """
        if filter_string == '':
            return data
        returned_data = []
        try:
            int(filter_string[1:])
        except ValueError:
            return data
        except IndexError:
            return data
        except TypeError:
            return data
        else:
            if filter_string[0] == 'G' and len(filter_string) <= 4:
                for call in data:
                    if call.duration > int(filter_string[1:]):
                        returned_data.append(call)
            elif filter_string[0] == "L" and len(filter_string) <= 4:
                for call in data:
                    if call.duration < int(filter_string[1:]):
                        returned_data.append(call)
            else:
                return data
            return returned_data

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        return "Filter calls based on duration; " \
               "L### returns calls less than specified length, G### for greater"


class LocationFilter(Filter):
    """
    A class for selecting only the calls that took place within a specific area
    """
    def apply(self, customers: list[Customer],
              data: list[Call],
              filter_string: str) \
            -> list[Call]:
        """ Return a list of all unique calls from <data>, which took
        place within a location specified by the <filter_string>
        (at least the source or the destination of the event was
        in the range of coordinates from the <filter_string>).

        The <customers> list contains all customers from the input dataset.

        The filter string is valid if and only if it contains four valid
        coordinates within the map boundaries.
        These coordinates represent the location of the lower left corner
        and the upper right corner of the search location rectangle,
        as 2 pairs of longitude/latitude coordinates, each separated by
        a comma and a space:
          lowerLong, lowerLat, upperLong, upperLat
        Calls that fall exactly on the boundary of this rectangle are
        considered a match as well.
        - If the filter string is invalid, return the original list <data>
        - If the filter string is invalid, your code must not crash, as
        specified in the handout.

        Do not mutate any of the function argumennts!
        """
        if filter_string == '':
            return data
        returned_data = []
        try:
            lowerlong, lowerlat, upperlong, upperlat = filter_string.split(', ')
            float(lowerlong)
            float(lowerlat)
            float(upperlong)
            float(upperlat)
        except ValueError:
            return data
        except AttributeError:
            return data
        lowerlong, lowerlat, upperlong, upperlat = filter_string.split(', ')
        lowerlong = float(lowerlong)
        lowerlat = float(lowerlat)
        upperlong = float(upperlong)
        upperlat = float(upperlat)
        if lowerlong < -79.697878 or lowerlong > -79.196382:
            return data
        elif lowerlat < 43.576959 or lowerlat > 43.799568:
            return data
        elif upperlong > -79.196382 or upperlong < -79.697878:
            return data
        elif upperlat > 43.799568 or upperlat < 43.576959:
            return data
        for call in data:
            if lowerlong <= call.dst_loc[0] <= upperlong \
                    and lowerlat <= call.dst_loc[1] <= upperlat:
                returned_data.append(call)
            elif lowerlong <= call.src_loc[0] <= upperlong \
                    and lowerlat <= call.src_loc[1] <= upperlat:
                returned_data.append(call)
        return returned_data

    def __str__(self) -> str:
        """ Return a description of this filter to be displayed in the UI menu
        """
        return "Filter calls made or received in a given rectangular area. " \
               "Format: \"lowerLong, lowerLat, " \
               "upperLong, upperLat\" (e.g., -79.6, 43.6, -79.3, 43.7)"


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'allowed-import-modules': [
            'python_ta', 'typing', 'time', 'datetime', 'call', 'customer'
        ],
        'max-nested-blocks': 4,
        'allowed-io': ['apply', '__str__'],
        'disable': ['W0611', 'W0703'],
        'generated-members': 'pygame.*'
    })
