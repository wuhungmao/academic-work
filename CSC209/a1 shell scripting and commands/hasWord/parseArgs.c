#include <stdio.h>
#include "myStrings.h"
#include "parseArgs.h"

int iWord=-1, iFileName=-1;

/**
 * print the usage message
 */
void usage(){
	fprintf(stderr, "usage: parseArgs [--help] word filename\n");
}

/**
 * print error message to stderr
 */
void optionsError(char mesg[]){
	fprintf(stderr, "hasWord: %s\nTry 'hasWord --help' for more information.\n", mesg);
}

/*
 * parse and understand the args. 
 * returns 1 for success, 0 for error, 2 for --help printed
 * On success, iWord, iFileName are all properly initialized
 * iFileName == -1 indicates read from stdin
 */
int parseArgs(int argc, char *argv[]){
	if (argc == 1 || argc >= 4) {
		fprintf(stderr, "hasWord: incorrect number of arguments.\nTry 'hasWord --help' for more information.\n");
		iWord = -1;
		iFileName = -1;
		return 0;
	}
	else if (argc == 2){
		iWord = 1;
		iFileName = -1;
		int i = 0;
		while (argc != 0) {
			if (myisStrEQ(argv[i], "--help")) {
				usage();
				return 2;
				break;
			}
			i++;
			argc--;
		}
		return 1;
	}
	else if (argc == 3){
		iWord = 1;
		iFileName = 2;
		int i = 0;
		while (argc != 0) {
			if (myisStrEQ(argv[i], "--help")) {
				usage();
				return 2;
				break;
			}
			i++;
			argc--;
		}
		return 1;
	}
}
