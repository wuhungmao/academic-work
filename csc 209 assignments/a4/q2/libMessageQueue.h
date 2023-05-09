#include "protocol.h"

#define MAX_MESSAGE_QUEUE_LEN 20
typedef struct MessageQueue {
	char message[MAX_MESSAGE_QUEUE_LEN][MAX_MESSAGE_LEN];
	int len; // the number of things currently in the queue
	int capacity;
} MessageQueue;

/**
 * initialize a message queue
 */
void initQueue(MessageQueue *queue);

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
int enqueue(MessageQueue *queue, char *message);

/**
 * dequeue the top of the queue
 * params: 
 * 	MessageQueue queue: the queue we are dequeueing
 * 	char *message: strlen(message)<MAX_MESSAGE_LEN
 * return: 
 * 	1 if dequeud
 * 	0 otherwise
 */
int dequeue(MessageQueue *queue, char *message);
