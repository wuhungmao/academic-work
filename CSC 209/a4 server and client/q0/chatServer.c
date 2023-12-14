/* server process */

/* include the necessary header files */
#include<ctype.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<stdlib.h>
#include<arpa/inet.h>
#include<stdio.h>
#include<unistd.h>
#include<string.h>

#include "protocol.h"
#include "libParseMessage.h"
#include "libMessageQueue.h"

/**
 * send a single message to client 
 * sockfd: the socket to read from
 * toClient: a buffer containing a null terminated string with length at most 
 * 	     MAX_MESSAGE_LEN-1 characters. We send the message with \n replacing \0
 * 	     for a mximmum message sent of length MAX_MESSAGE_LEN (including \n).
 * return 1, if we have successfully sent the message
 * return 2, if we could not write the message
 */
int sendMessage(int sfd, char *toClient){
	char c;
	int offset = 0;
	while (1){
		c = toClient[offset];
		if(c=='\0')c='\n';
		int numSend = send(sfd, &c, 1, 0);
		if(numSend!=1)return(2);
		if(c=='\n')break;
		offset+=1;
	}
	return(1);
}

/**
 * read a single message from the client. 
 * sockfd: the socket to read from
 * fromClient: a buffer of MAX_MESSAGE_LEN characters to place the resulting message
 *             the message is converted from newline to null terminated, 
 *             that is the trailing \n is replaced with \0
 * return 1, if we have received a newline terminated string
 * return 2, if the socket closed (read returned 0 characters)
 * return 3, if we have read more bytes than allowed for a message by the protocol
 */
int recvMessage(int sfd, char *fromClient){
	char c;
	int len= 0;
	while (1){
		if(len==MAX_MESSAGE_LEN)return(3);

		int numRecv = recv(sfd, &c, 1, 0);
		if(numRecv==0)return(2);
		if(c=='\n')c='\0';
		fromClient[len]=c;
		if(c=='\0')return(1);
		len+=1;
	}
}

int main (int argc, char ** argv) {
    int sockfd;

    if(argc!=2){
	    fprintf(stderr, "Usage: %s portNumber\n", argv[0]);
	    exit(1);
    }
    int port = atoi(argv[1]);

    if ((sockfd = socket (AF_INET, SOCK_STREAM, 0)) == -1) {
        perror ("socket call failed");
        exit (1);
    }

    struct sockaddr_in server;
    server.sin_family=AF_INET;          // IPv4 address
    server.sin_addr.s_addr=INADDR_ANY;  // Allow use of any interface 
    server.sin_port = htons(port);      // specify port

    if (bind (sockfd, (struct sockaddr *) &server, sizeof(server)) == -1) {
        perror ("bind call failed");
        exit (1);
    }

    if (listen (sockfd, 5) == -1) {
        perror ("listen call failed");
        exit (1);
    }

    for (;;) {
	int newsockfd;
        if ((newsockfd = accept (sockfd, NULL, NULL)) == -1) {
            perror ("accept call failed");
            continue;
        }

        if (fork () == 0) {
		char user[MAX_USER_LEN+1]; 
		user[0]='\0';

		MessageQueue queue;
		initQueue(&queue);

	    	char fromClient[MAX_MESSAGE_LEN], toClient[MAX_MESSAGE_LEN];

		while(1){
	    		int retVal=recvMessage(newsockfd, fromClient); 
	    		if(retVal==1){
				// we have a null terminated string from the client
				char *part[4];
				int numParts=parseMessage(fromClient, part);
				if(numParts==0){
					strcpy(toClient,"ERROR");
					sendMessage(newsockfd, toClient);
				} else if(strcmp(part[0], "list")==0){
					sprintf(toClient, "users:%s",user);
					sendMessage(newsockfd, toClient);
				} else if(strcmp(part[0], "message")==0){
					char *fromUser=part[1];
					char *toUser=part[2];
					char *message=part[3];

					if(strcmp(fromUser, user)!=0){
						sprintf(toClient, "invalidFromUser:%s",fromUser);
						sendMessage(newsockfd, toClient);
					} else if(strcmp(toUser, user)!=0){ // Right now we can only send messages to ourselves!
						sprintf(toClient, "invalidToUser:%s",toUser);
						sendMessage(newsockfd, toClient);
					} else {
						sprintf(toClient, "%s:%s:%s:%s","message", fromUser, toUser, message);
						if(enqueue(&queue, toClient)){
							strcpy(toClient, "messageQueued");
							sendMessage(newsockfd, toClient);
						}else{
							strcpy(toClient, "messageNotQueued");
							sendMessage(newsockfd, toClient);
						}
					}
				} else if(strcmp(part[0], "quit")==0){
					strcpy(toClient, "closing");
					sendMessage(newsockfd, toClient);
            				close (newsockfd);
            				exit (0);
				} else if(strcmp(part[0], "getMessage")==0){
					if(dequeue(&queue, toClient)){
						sendMessage(newsockfd, toClient);
					} else {
						strcpy(toClient, "noMessage");
						sendMessage(newsockfd, toClient);
					}
				} else if(strcmp(part[0], "register")==0){
					if(strncmp(user, part[1], MAX_USER_LEN)!=0){
						strcpy(user, part[1]);
						strcpy(toClient, "registered");
						sendMessage(newsockfd, toClient);
					} else {
						strcpy(toClient, "userAlreadyRegistered");
						sendMessage(newsockfd, toClient);
					}
				}
	    		} else {
            			close (newsockfd);
            			exit (0);
	    		}
		}
        }
        close (newsockfd);
    }
    exit(0);
}
