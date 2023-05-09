test_libDict: test_libDict.c libDict.h libDict.o
	gcc -g -o test_libDict test_libDict.c libDict.o

libDict.o: libDict.c libDict.h
	gcc -g -c libDict.c

clean:
	rm -f *.o test_libDict
