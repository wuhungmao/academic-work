all: parallelDo testLib isPrime

clean:
	rm -f parallelDo libProcessControl.o libParseArgs.o testLib isPrime

libProcessControl.o: libProcessControl.c libProcessControl.h
	gcc -g -c libProcessControl.c

libParseArgs.o: libParseArgs.c libParseArgs.h
	gcc -g -c libParseArgs.c

testLib: testLib.c libProcessControl.o libParseArgs.o
	gcc -g -o testLib testLib.c libProcessControl.o libParseArgs.o

parallelDo: parallelDo.c libProcessControl.o libParseArgs.o
	gcc -g -o parallelDo parallelDo.c libProcessControl.o libParseArgs.o
