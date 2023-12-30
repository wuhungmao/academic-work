#include <stdio.h>
#include <stdlib.h>
#include "myStrings.h"
#include "parseArgs.h"

/*
return whether word appears in open fpin
 */
int isWordInFile(char word[], int wordLen, FILE *fpin){
	int filenum = 0;
	char c[1000];
	int i=0;
	c[0]=fgetc(fpin);
	while (c[i] != EOF) {
		i++;
		c[i]=fgetc(fpin);
		filenum++;
	}
	c[-1]='\n';
	int j=0;
	while(1){
		int hasword=1;
		for (int i = 0; i < wordLen; i++){
			if (c[i+j] != word[i]) {
				hasword = 0;
				break;
			}
		}
		if (hasword == 1) {
			return hasword;
		}
		j++;
		if(filenum==j){
			break;
		}
	}
	return 0;
}

int main(int argc, char *argv[]) {
	FILE *fpin;
	int retVal=parseArgs(argc, argv);
	if (retVal == 0) {
		exit(127);
	}else if (retVal == 1 && iFileName != -1){
		fpin = fopen(argv[iFileName], "r");
		int result = isWordInFile(argv[iWord], myStrLen(argv[iWord]), fpin);
		fclose(fpin);
		if (result == 1){
			exit(0);
		} else if (result == 0) {
			exit(1);
		}
	}else if (retVal == 1 && iFileName == -1){
		fpin = stdin;
		int result = isWordInFile(argv[iWord], myStrLen(argv[iWord]),fpin);
		if (result == 1){
			exit(0);
		} else if (result == 0) {
			exit(1);
		}
	} else if (retVal == 2){
		exit(0);
	}
}
