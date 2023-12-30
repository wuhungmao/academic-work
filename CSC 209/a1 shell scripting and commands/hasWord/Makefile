all: hasWord test_parseArgs

hasWord: hasWord.o parseArgs.o myStrings.o
	gcc -o hasWord hasWord.o parseArgs.o myStrings.o

hasWord.o: hasWord.c myStrings.h parseArgs.h
	gcc -c hasWord.c

myStrings.o: myStrings.h myStrings.c
	gcc -c myStrings.c

parseArgs.o: parseArgs.h parseArgs.c myStrings.h
	gcc -c parseArgs.c

test_parseArgs: test_parseArgs.o parseArgs.o myStrings.o parseArgs.h
	gcc -o test_parseArgs test_parseArgs.o parseArgs.o myStrings.o

test_parseArgs.o: test_parseArgs.c parseArgs.h
	gcc -c test_parseArgs.c
		
clean:
	rm -f *.o test_parseArgs hasWord
