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

#include<sys/select.h>

#include "protocol.h"
#include "libParseMessage.h"
#include "libMessageQueue.h"

typedef struct message_queue_lst {
	int fd;
	char *user_name;
	MessageQueue queue;
	char buffer[5*MAX_MESSAGE_LEN];
	char input_buffer[MAX_MESSAGE_LEN];
} message_queue_lst;

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
	int len;
	len = strlen(toClient);
	if(toClient[len-1] == '\0'){
		toClient[len-1]='\n';
	}
	int numSend = send(sfd, toClient, len, 0);
	if(numSend==-1)return(2);
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
 * return 4, if server receive incomplete message
 */
int recvMessage(int sfd, char *fromClient, int fd, int *num_mess){
	/*find length of text*/
	/*
	int numRecv = recv(sfd, fromClient, MAX_MESSAGE_LEN, 0);
	if((numRecv == MAX_MESSAGE_LEN) && (fromClient[numRecv-1] !='\n')) {
	return(3);\
	} else if(numRecv==0) {
	return(2);
	} else {
		fromClient[numRecv-1]='\0';
		return(1);
	}*/
	int len_input = strlen(fromClient);
	char input_buffer[MAX_MESSAGE_LEN - len_input];
	int numRecv = recv(sfd, input_buffer, MAX_MESSAGE_LEN - len_input, 0);
	if(((numRecv + len_input)==MAX_MESSAGE_LEN) && (input_buffer[numRecv-1] != '\n'))return(3);
	if(numRecv==0)return(2);
	strncat(fromClient, input_buffer, numRecv);
	int found = 0;
	for (int i = 0; i <strlen(fromClient); i++) {
		if (fromClient[i] == '\n') {
			found=1;
			*num_mess = *num_mess + 1;
		}
	}
	if(found == 1){
		return(1);
	}else{
		return(4);
	}
}

int max(int x, int y) {
	if (x>y) {
		return x;
	} else {
		return y;
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

	/*
	fdlist contains 32 client sockets and 1 listening socket
	first socket stored in fdlist is sockfd for listening new socket
	*/
	int fdlist[33];
	for (int i = 1; i< 33; i++) {
		fdlist[i] = -1;
	}
	fdlist[0]=sockfd;
	message_queue_lst *client_mess_lst;
	client_mess_lst = (message_queue_lst *) malloc(sizeof(message_queue_lst)*1024);
	for (int i = 0; i<1024; i++) {
		client_mess_lst[i].user_name = "";
	}
	char* user_lst[32];
	for (int i = 0; i< 32; i++) {
		user_lst[i] = "";
	}

	/*ideas*/
		/*each client has a mess queue. In each loop, the algorithm reads ready fd one by one, if the fd
		decides to sends message to another client, process it and put message to mess queue of 
		that client. if the fd decides to retrieve message and "getMessage" is in fd, recvMessage on that fd 
		and check if chat message is empty, send "no message" to that fd in writefds, if not,
		take one message out and send it to client.
		*/

	/*In a loop*/
	/*prepare for select call*/
	/*if first round -> no client socket*/
		/*add listening socket to fd_lst so server can accept new socket*/
	/*if second+ round -> at least one client socket*/
		/*add listening socket to fd_lst so server can accept new socket*/
		/*After getting new client sockets from first round, add it to readfds and writefds*/
    for (;;) {

		fd_set readfds, writefds, exceptfds;
		FD_ZERO(&readfds);                 
		FD_ZERO(&writefds);
        FD_ZERO(&exceptfds);   

		/*select shouldn't wait for any fds, we are in a loop*/
		struct timeval tv;
        tv.tv_sec=0;          
        tv.tv_usec=0;
		int fdmax=0;    
		/*add all fd to writefds and calculate maximum fd for select call at the end after reading and processing
		*/                          
        for (int i=0; i<33; i++) {
            if (fdlist[i]>0) {
                FD_SET(fdlist[i], &writefds);  // poke the fdlist[i] bit of readfds
                fdmax=max(fdmax,fdlist[i]);
            }
        }
		int numfds;
		if ((numfds=select(fdmax+1, NULL, &writefds, NULL, &tv))>0) {
			for (int i=0; i<33; i++) {
                if (FD_ISSET(fdlist[i],&writefds)) {
					if (strlen(client_mess_lst[fdlist[i]].buffer) != 0) {
						sendMessage(fdlist[i], client_mess_lst[fdlist[i]].buffer);
						client_mess_lst[fdlist[i]].buffer[0] = '\0';
					}
				}
			}
		}

		/*add all fd to readfds and calculate maximum fd for select call later
		*/
		                      
        for (int i=0; i<33; i++) {
            if (fdlist[i]>0) {
                FD_SET(fdlist[i], &readfds);  // poke the fdlist[i] bit of readfds
                fdmax=max(fdmax,fdlist[i]);
            }
        }

		/*numfds is used to stored number of files that are ready to read*/
		
		if ((numfds=select(fdmax+1, &readfds, NULL, NULL, &tv))>0) {
			/*accept new socket*/
			for (int i=0; i<33; i++) {
                if (FD_ISSET(fdlist[i],&readfds)) {
                    if (fdlist[i]==sockfd) {/*accept new client*/
                        int newsockfd;
                        if ((newsockfd = accept (sockfd, NULL, NULL)) == -1) {
                            perror ("accept call failed");
                            continue;
                        }
						/*insert new socket fd either at a position where fd's been removed or at the end of fd list*/
						int full = 1;
						for (int k = 1; k<33; k++) {
							if (fdlist[k] == -1) {
								fdlist[k]=newsockfd;
								full = 0;
								break;
							}
						}
						if (full == 1) {
							close(newsockfd);
							continue;
						}
						/*initialize message queue for new client and add it to message list*/
						MessageQueue queue;
						initQueue(&queue);
						client_mess_lst[newsockfd].fd = newsockfd;
						client_mess_lst[newsockfd].queue = queue;
					} else { 
						/*fdlist[i] is ready and it is not listening socket*/
						/*recvMessage(fdlist[i], fromClient) */
							char toClient[MAX_MESSAGE_LEN];
							int num_mess = 0;
							int retVal=recvMessage(fdlist[i], client_mess_lst[fdlist[i]].input_buffer, fdlist[i], &num_mess);
							if(retVal==1){
								// we have at least one null terminated string from the client
								char *mess;
								char delim[] = "\n";
								mess = strtok(client_mess_lst[fdlist[i]].input_buffer, delim);
								for (int mess_num = 1; mess_num < num_mess+1; mess_num++) {
									char *part[4];
									int numParts = 0;
									if (mess != NULL) {
										numParts=parseMessage(mess, part);
									}
									if(numParts==0){
										strcpy(toClient,"ERROR\n");
										strcat(client_mess_lst[fdlist[i]].buffer, toClient);
									} else if(strcmp(part[0], "list")==0){
										/*concatenate every user name in user_lst to user, then send users name to client*/
										char user[18*MAX_USER_LEN] = "";
										int count = 0;
										int count_user = 0;
										for (int j = 0; j < 32; j++) {
											for(int k = 0; k < (int) strlen(user_lst[j]); k++) {
												user[count]=user_lst[j][k];
												count++;
											}
											if (strlen(user_lst[j]) != 0) {
												count_user++;
												user[count]=' ';
												count++;
											}
											if (count_user == 10) {
												break;
											}
										}
										sprintf(toClient, "users:%s\n",user);
										strcat(client_mess_lst[fdlist[i]].buffer, toClient);
									} else if(strcmp(part[0], "message")==0){

										char *fromUser=part[1];
										char *toUser=part[2];
										char *message=part[3];

										/*attempt to find toUser in user_lst, if not found, toUser does not exist.*/
										int found = 0;
										for (int j = 0; j<32; j++){
											if(strcmp(toUser, user_lst[j])==0){
												found = 1;
												break;
											}
										}
										/*compare sender name in message with its username to make sure they are same person*/
										if((strcmp(fromUser, client_mess_lst[fdlist[i]].user_name)!=0) || (strlen(fromUser) == 0)){
											sprintf(toClient, "invalidFromUser:%s\n",fromUser);
											strcat(client_mess_lst[fdlist[i]].buffer, toClient);
										} else if((found == 0) || (strlen(toUser) == 0)) {
											/*cannot find receiver*/
											sprintf(toClient, "invalidToUser:%s\n",toUser);
											strcat(client_mess_lst[fdlist[i]].buffer, toClient);
										} else {
											/*both sender and receiver name are valid. Find message queue for receiver in the for loop
											and enqueue message. If fails to enqueue message, then send "messageNotQueued" to sender*/
											sprintf(toClient, "%s:%s:%s:%s\n","message", fromUser, toUser, message);
											
											for (int fd = 1; fd<1024; fd++) {
												if (strcmp(client_mess_lst[fd].user_name, toUser) == 0) {
													(client_mess_lst[fd].queue).capacity=20;
													if (enqueue(&(client_mess_lst[fd].queue), toClient)) {
														strcpy(toClient, "messageQueued\n");
														strcat(client_mess_lst[fdlist[i]].buffer, toClient);
														break;
													} else {
														strcpy(toClient, "messageNotQueued\n");
														strcat(client_mess_lst[fdlist[i]].buffer, toClient);
														break;
													}
												}
											}
										}
										
									} else if(strcmp(part[0], "quit")==0){
										strcpy(toClient, "closingConnection\n");
										sendMessage(fdlist[i], toClient);
										for (int k = 0; k<32; k++) {
											if (strcmp(user_lst[k], client_mess_lst[fdlist[i]].user_name) == 0) {
												user_lst[k] = "";
												break;
											}
										}
										client_mess_lst[fdlist[i]].fd = -1;
										client_mess_lst[fdlist[i]].user_name = "";
										strcpy(client_mess_lst[fdlist[i]].input_buffer, "");
										close(fdlist[i]);
										fdlist[i]=-1;
										continue;
									} else if(strcmp(part[0], "getMessage")==0){
										if(dequeue(&(client_mess_lst[fdlist[i]].queue), toClient)){
											strcat(client_mess_lst[fdlist[i]].buffer, toClient);
										} else {
											strcpy(toClient, "noMessage\n");
											strcat(client_mess_lst[fdlist[i]].buffer, toClient);
										}
									} else if(strcmp(part[0], "register")==0){
										int found=0;
										/*check if user already registers*/
										if (strlen(client_mess_lst[fdlist[i]].user_name) != 0) {
											strcpy(toClient, "ERROR\n");
											strcat(client_mess_lst[fdlist[i]].buffer, toClient);
											found = -1;
										}
										/*add user to user_lst so user_lst keep track of all client user name*/
										if (found == 0) {
											for (int j = 0; j < 32; j++) {
												if(strcmp(part[1], user_lst[j])==0){
													strcpy(toClient, "userAlreadyRegistered\n");
													strcat(client_mess_lst[fdlist[i]].buffer, toClient);
													found = 1;
													break;
												}
											}
											if (found == 0) {
												client_mess_lst[fdlist[i]].user_name = strdup(part[1]);
												for (int k = 0;k < 32; k++) {
													if (strcmp(user_lst[k], "") == 0) {
														user_lst[k] = strdup(part[1]);
														break;
													}
												}
												sprintf(toClient, "registered:%s\n", part[1]);
												strcat(client_mess_lst[fdlist[i]].buffer, toClient);
											}
										}
									}
									client_mess_lst[fdlist[i]].input_buffer[0] = '\0';
									mess = strtok(NULL, delim);
								}
							} else if (retVal == 4) {
								/*we receive an incomplete message, we put the message in a buffer already, so we can
								continue and deal with it later. Let server handles other clients first*/
								continue;
							} else {
								/*!!! make sure to clean up everything if socket closed on user's end*/
								strcpy(toClient, "closingConnection\n");
								sendMessage(fdlist[i], toClient);
								for (int k = 0; k<32; k++) {
									if (strcmp(user_lst[k], client_mess_lst[fdlist[i]].user_name) == 0) {
										user_lst[k] = "";
										break;
									}
								}
								client_mess_lst[fdlist[i]].fd = -1;
								client_mess_lst[fdlist[i]].user_name = "";
								close(fdlist[i]);
								fdlist[i]=-1;
								strcpy(client_mess_lst[fdlist[i]].input_buffer, "");
							}
					}
				}
			}
		}
	}
}
	
