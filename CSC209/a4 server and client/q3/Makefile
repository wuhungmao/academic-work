all: chatServer libMessageQueue.o libParseMessage.o test_libParseMessage 

clean:
	rm -f chatServer libMessageQueue.o libParseMessage.o test_libParseMessage 

chatServer: chatServer.c libParseMessage.o libMessageQueue.o protocol.h
	gcc -g -Wall -o chatServer chatServer.c libParseMessage.o libMessageQueue.o
	
test_libParseMessage: libParseMessage.o test_libParseMessage.c libParseMessage.h protocol.h
	gcc -g -Wall -o test_libParseMessage test_libParseMessage.c libParseMessage.o

libParseMessage.o: libParseMessage.c libParseMessage.h protocol.h
	gcc -g -Wall -c libParseMessage.c

libMessageQueue.o: libMessageQueue.c libMessageQueue.h protocol.h
	gcc -g -Wall -c libMessageQueue.c 
