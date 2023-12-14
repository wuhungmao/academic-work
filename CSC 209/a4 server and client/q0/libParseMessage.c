#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "libParseMessage.h"
#include "protocol.h"

int isAlphaNumeric(char *s){
	while(*s!='\0'){
	       if(!isalnum(*s))return(0);
	       s++;
	}
	return(1);
}
void clearPart(char *part[]){
	for(int i=0;i<4;i++)part[i]=NULL;
}

/**
 * Split buffer into at most 4 null terminated parts, 
 * part[0]...part[3] by placing '\0' inside buffer replacing the
 * first 3 ':'. We return pointers to the pieces. We also
 * validate the message at a high level.
 *
 * params
 * 	char *buffer: a null terminated string
 * 	char *part[]: array of 4 pointers to char
 * return
 * 	the number of pieces parsed
 * 	0 if this is not a valid message
 */
int parseMessage(char *buffer, char *part[]){
	clearPart(part);

	char *s=buffer;
	int numParts=0;
	part[0]=buffer;
	numParts++;
	while(*s!='\0' && numParts<4){
		if(*s==':'){
			*s='\0';
			part[numParts]=(s+1);
			numParts++;
		}
		s++;
	}

	if(strcmp(part[0], "register")==0){ // register:USER
		if(numParts!=2 || strlen(part[1])>MAX_USER_LEN){
			clearPart(part);
			return(0);
		}
	} else if(strcmp(part[0], "getMessage")==0){ // getMessage
		if(numParts!=1){
			clearPart(part);
			return(0);
		}
	} else if(strcmp(part[0], "list")==0){ // list
		if(numParts!=1){
			clearPart(part);
			return(0);
		}
	} else if(strcmp(part[0], "quit")==0){ // quit
		if(numParts!=1){
			clearPart(part);
			return(0);
		}
	} else if(strcmp(part[0], "message")==0){ // message:FROM_USER:TO_USER:CHAT_MESSAGE
		if(numParts!=4 
			|| strlen(part[1])>MAX_USER_LEN || !isAlphaNumeric(part[1])
			|| strlen(part[2])>MAX_USER_LEN || !isAlphaNumeric(part[2])
			|| strlen(part[3])>MAX_CHAT_MESSAGE_LEN){
				clearPart(part);
				return(0);
		}
	} else {
		clearPart(part);
		return(0);
	}
	return(numParts);
}

