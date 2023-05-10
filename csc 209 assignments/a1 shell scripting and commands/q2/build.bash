#!/bin/bash
# The following is a script to build everything in this directory
# BUT instead you can just use 
# make clean
# make myStrings
# make hasWord
# make test_parseArgs

gcc -c myStrings.c

gcc -c parseArgs.c
gcc -c test_parseArgs.c 
gcc -o test_parseArgs test_parseArgs.o parseArgs.o myStrings.o

gcc -c hasWord.c
gcc -o hasWord hasWord.o myStrings.o parseArgs.o
