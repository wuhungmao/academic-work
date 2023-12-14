#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "libParseArgs.h"

/**
 * parallelDo -n NUM -o OUTPUT_DIR COMMAND_TEMPLATE ::: [ ARGUMENT_LIST ...]
 * build and execute shell command lines from standard input in parallel
 */

PARALLEL_PARAMS pparams;

void printParallelParams(){
	printf("maxNumRunning=%d\n", pparams.maxNumRunning);
	printf("outputDir=%s\n", pparams.outputDir);

	printf("commandTemplate=");
	printf("%s\n", pparams.commandTemplate);

	printf("argumentList=");
	for(int i=0;i<pparams.argumentListLen;i++)
		printf("%s ", pparams.argumentList[i]);
	printf("\n");
}

int parseArgs(int argc, char *argv[]){
	pparams.maxNumRunning=0;
	pparams.outputDir=NULL;
	pparams.commandTemplate = NULL;
	pparams.argumentList=NULL;
	pparams.argumentListLen=0;

	if(argc<6)return(0);
	if(strncmp(argv[1], "-n", 3)!=0)return(0);
	pparams.maxNumRunning = atoi(argv[2]); /*max num of command to run in parallel?*/

	if(strncmp(argv[3], "-o", 3)!=0)return(0);
	pparams.outputDir = argv[4];

	pparams.commandTemplate = argv[5];
	if(argc==6)return(1);

	if(strcmp(argv[6],":::")!=0)return(0); /*???*/

	pparams.argumentList=&argv[7];
	pparams.argumentListLen=argc-7;

	return(1);
}

