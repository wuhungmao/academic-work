"""
This task can be done on your machine or the lab machine (no VM required).

Hashing takes an infinite input space and maps it to a finite space.
Collisions are guaranteed to exist. 

A few years ago, Google spend millions of dollars proving that collisions
exist for a very famous hashing algorithm. 

While this isn't the best approach to brute force an algorithm, it does work. 

You will be finding different inputs that (may) generate a hash collision, thereby testing whether our custom hashing scheme and custom input universe will generate hash collisions. 

Learning outcome:
A) Probably not a good idea to come up with your own hashing algorithm, unless you are very good at it.
B) Vastly different inputs can generally generate the same hash.

Below is the starter code, complete the section that says "YOUR CODE HERE"
and answer the following questions:

0) [1 point]  How many different hashes exist in this custom hashing scheme (hashthis()) 
md5 hashes are 32 hexdecimal characters long, so md5sum[25:] is 7 characters long. Therefore there are 16^7 different hashes exist in hashthis(). 

1) [1 point] 
1a) Theoretically, how many inputs are possible 
(look at the alphabet and length of strings you need to generate, among ONLY those strings, is a cash collision possible within our custom hashing scheme (not considering string s2 at all, but only considering the alphabet and the strings you are generating)).
(show your work)
There are 26 length-1 strings, 26^2 length-2 strings, ..., 26^10 length-10 strings. 
Therefore, theoretically there are 26 + 26^2 + ... + 26^10 = (26^11 - 26) / 25 inputs (geometric series)

1b) Given your answer for 0) and 1a) does this mean that there are guaranteed to be hash collisions? 
(Look at the alphabet and length of strings you need to generate, among ONLY those strings, is a cash collision possible within our custom hashing scheme (not considering string s2 at all, but only considering the alphabet and the strings you are generating)).
(show your work)
(yes, it is guaranteed to have a collision; no, it is possible that there will be no collisions).

Yes
There are 16^7 = 268,435,456 hashes. 
There are (26^11 - 26) / 25 = 146,813,779,479,510 possible inputs. 
Since there are way more possible inputs than possible hashes, by Pigeon-hole theorem, it is guaranteed to have a collision. 


2) [1 point] Which string caused a hash collision (if any) with the string "CSC347is_very_awesome" but differs from this string, i.e., do not test CSC347is_very_awesome vs CSC347is_very_awesome? (name one or answer "no collision detected". 
(note your code must complete within 2 minutes, it should take about 0-30 seconds in an ideal environment)
avghwx


3) [1 point] Complete the following command (this is what the TAs will run to give you points) 
python3 hashcollisions_starter.py -a CSC347is_very_awesome -b avghwx   
(note your code must produce the same output/result as our sample solution)
"""



import itertools
import getopt
import sys
import hashlib

def hashthis(plain):
  """ Do not change this function"""
  # To keep it simple, create an MD5 hash object
  md5_hash = hashlib.md5()

  # Update the hash object with the input string
  md5_hash.update(plain.encode('utf-8'))

  # Get the hexadecimal representation of the hash
  md5sum = md5_hash.hexdigest()

  return md5sum[25:]  # we will take a chunk of it.

def generate_strings():
  """
  Implement this function. 
  Its signature is: def generate_strings() -> Iterator[str]:
  """
  alphabet = 'abcdefghijklmnopqrstuvwxyz'  # Lowercase letters
 
  ### TODO: YOUR CODE HERE
  # Generate all possible string of length [1 to  10] (inclusive)
  # and test them to find if there exists a collision with 
  # CSC347is_very_awesome
  # Generate lengths from 1 to 10 
  strings = []
  for char in alphabet: 
     strings.append(char)
     yield char
  
  for x in range(2,11):
     new = []
     for string in strings:
        for char in alphabet: 
           new.append(string + char)
           yield string + char

     strings = new

def find_collision():
  """ Do not change this function"""
  s1 = ""
  s2 = "CSC347is_very_awesome"
  h1 = hashthis(s1)
  h2 = hashthis(s2)
  counter = 0
  for generated_string in generate_strings():
    h1 = hashthis(generated_string)
    counter += 1
    if counter % 1000 == 0:
      print(counter)
    if h1 == h2 and s2 != generated_string:
      break
  print(h1,"|", h2,"|", generated_string,"|", s2)

  return (generated_string, s2)         



def main(argv):
    """ Do not change this function"""
    arg1 = None
    arg2 = None

    try:
        opts, args = getopt.getopt(argv, "a:b:", ["arg1=", "arg2="])
    except getopt.GetoptError:
        print("Usage: python program.py -a <arg1> -b <arg2>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-a", "--arg1"):
            arg1 = arg
        elif opt in ("-b", "--arg2"):
            arg2 = arg
    if arg1 == "CSC347is_very_awesome": 
        print(hashthis(arg1))
        print(hashthis(arg2))
        sys.exit(0) 
    if arg1 and arg2:
        print("Argument 1:", arg1)
        print("Argument 2:", arg2)
        print(hashthis(arg1))
        print(hashthis(arg2))
      
        s1, s2 = find_collision()        
        print(hashthis(s1))
        print(hashthis(s2))
  
    else:
        print("Usage: python program.py -a <arg1> -b <arg2>")

if __name__ == "__main__":
  """ Do not change this function"""
  main(sys.argv[1:])
