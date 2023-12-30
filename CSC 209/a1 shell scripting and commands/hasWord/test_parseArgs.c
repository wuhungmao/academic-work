#include <stdio.h>
#include <stdlib.h>
#include "parseArgs.h"

void printArrayOfString(int len, char *s[]){
	printf("[");
	for(int i=0;i<len;i++){
		printf("%s,",s[i]);
	}
	printf("]");
}

void check(int argc, char *argv[], int eRetVal, int eiWord, int eiFileName){
	int retVal = 0;
	char *mess;
	char *ok="OK ";
	char *err="ERR";

	retVal = parseArgs(argc, argv);

	if(retVal==eRetVal && iWord==eiWord && iFileName==eiFileName){
		mess=ok;
	} else {
		mess=err;
	}
	printf("%s (return, iWord,iFileName) = (%d,%d,%d) EXPECTED (%d,%d,%d) parseArgs(%d,",mess,retVal, iWord, iFileName,eRetVal, eiWord, eiFileName, argc);
	printArrayOfString(argc, argv);
	printf(")\n");
}

int main(int argc, char *argv[]) {
	char *argv0[]= { };
	int argc0 = 0;

	char *argv1[]= { "./hasWord" };
	int argc1 = 1;

	char *argv2[]= { "./hasWord", "word" };
	int argc2 = 2;

	char *argv3[]= { "./hasWord", "word", "filename" };
	int argc3 = 3;

	char *argv4[]= { "./hasWord", "word", "filename", "other" };
	int argc4 = 4;

	char *argv5[]= { "./hasWord", "--help", "filename", "other" };
	int argc5 = 4;

	char *argv6[]= { "./hasWord", "word", "filename", "other", "--help", "other" };
	int argc6 = 6;

	check(argc0, argv0, 0, -1, -1);
	check(argc1, argv1, 0, -1, -1);
	check(argc2, argv2, 1, 1, -1);
	check(argc3, argv3, 1, 1, 2);
	check(argc4, argv4, 0, -1, -1);
	check(argc5, argv5, 2, -1, -1);
	check(argc6, argv6, 2, -1, -1);
	return(0);
}
