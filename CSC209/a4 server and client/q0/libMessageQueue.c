#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "libMessageQueue.h"

/**
 * initialize a message queue
 */
void initQueue(MessageQueue *queue){
	queue->len=0;
	queue->capacity=MAX_MESSAGE_QUEUE_LEN;
}

/**
 * Add message to queue, if there is space and this is
 * a valid message. 
 * params:
 * 	MessageQueue queue: the queue we are adding to
 * 	char *message: strlen(message)<MAX_MESSAGE_LEN
 *
 * return:
 * 	1 if queued
 * 	0 otherwise
 */
int enqueue(MessageQueue *queue, char *message){
	if(queue->len==queue->capacity)return(0);
	strncpy(queue->message[queue->len], message, MAX_MESSAGE_LEN);
	queue->message[queue->len][MAX_MESSAGE_LEN-1]='\0';
	queue->len++;
	return(1);
}
/**
 * dequeue the top of the queue
 * params: 
 * 	MessageQueue queue: the queue we are dequeueing
 * 	char *message: strlen(message)<MAX_MESSAGE_LEN
 * return: 
 * 	1 if dequeud
 * 	0 otherwise
 */
int dequeue(MessageQueue *queue, char *message){
	if(queue->len>0){
		strncpy(message, queue->message[0], MAX_MESSAGE_LEN);
		message[MAX_MESSAGE_LEN-1]='\0';
		queue->len--;
		// below is unnecessary, if you are a bit clever
		for(int i=0;i<queue->len;i++){
			strncpy(queue->message[i],queue->message[i+1], MAX_MESSAGE_LEN);
			queue->message[i][MAX_MESSAGE_LEN-1]='\0';
		}
		return(1);
	} else return(0);
}
