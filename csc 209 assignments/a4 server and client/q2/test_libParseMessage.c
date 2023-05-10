#include <stdio.h>
#include <string.h>
#include "libParseMessage.h"
#include "protocol.h"

int strEQ(char *s1, char*s2){
	if(s1==NULL || s2==NULL){
		if(s1==s2)return(1);
		return(0);
	} 
	if(strcmp(s1, s2)==0)return(1);
	return(0);
}

void test_parseMessage(int eRetVal, char *epart[], char *buffer){
	char *part[4];

	char buffer2[MAX_MESSAGE_LEN];
	// Note: since the following function alters the buffer, 
	// we had better place a copy someplace alterable
	strncpy(buffer2,buffer, MAX_MESSAGE_LEN);

	int retVal = parseMessage(buffer2, part);

	char *out;
	if(retVal == eRetVal)out="OK   ";
	if(retVal != eRetVal)out="ERROR";

	if(retVal==eRetVal){
		for(int i=0;i<4;i++){
			if(!strEQ(epart[i], part[i]))out="ERROR";
		}
	}

	printf("%s: EXPECTED: %d RETURNED: %d ",out, eRetVal, retVal);
	/**
	for(int i=0;i<4;i++){
		printf(" epart[%d]=%s part[%d]=%s", i, epart[i], i, part[i]);
	}
	**/
	printf(" parseMessage(%s)\n" ,buffer);
}

typedef struct TestCase {
	int eRetVal;
	char *buffer;
	char *epart[4];
} TestCase;

TestCase tc [] = {
	{ 0, "this is not a good message", {NULL, NULL, NULL, NULL}  },
	{ 1, "list", {"list", NULL, NULL, NULL}  },
	{ 2, "register:arnold", {  "register", "arnold", NULL, NULL } },
	{ 0, "register:arnold:somtheing", {  NULL, NULL, NULL, NULL } },
	{ 0, "register:arnold:somtheing:other", {  NULL, NULL, NULL, NULL } },
	{ 4, "message:arnold:sid:this is the message", {  "message", "arnold", "sid", "this is the message" } },
	{ 4, "message:arnold:sid:this is the message:and some more", {  "message", "arnold", "sid", "this is the message:and some more" } },
	{ 0, "message:arn old:sid:this is the message:and some more", {  NULL, NULL, NULL, NULL } },
	{ 0, NULL , {  NULL, NULL, NULL, NULL } }
};

int main(int argc, char ** argv){
	int i=0;
	while(tc[i].buffer!=NULL){
		test_parseMessage(tc[i].eRetVal, tc[i].epart, tc[i].buffer);
		i++;
	}
}
