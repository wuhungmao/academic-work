#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "libParseArgs.h"
#include "libProcessControl.h"

void usage(FILE *fp){
	char *s= "Usage: parallelDo -n NUM -o OUTPUT_DIR COMMAND_TEMPLATE ::: [ARGUMENT_LIST ... ] \n"
		"Build and execute shell command lines in parallel\n"
		"\n"
		"Wherever {} is seen in COMMAND_TEMPLATE, it is replaced by a member of ARGUMENT_LIST \n"
		"\n"
		"stdout is sent to OUTPUT_DIR/pid.stdout\n"
		"stderr is sent to OUTPUT_DIR/pid.stderr\n"
		"\n"
		"parallelDo accepts the following signals, with the corresponding actions\n"
		"\tSIGUSR1: print a brief status message to stdout\n\n"
		"\tNumJobs NumCompleted NumRunning\n\n"
		"\tSIGUSR2: print a full status message to stdout\n"
		"\tNumJobs NumCompleted NumRunning\n"
		"\tPid1 IfExited1 ExitStatus1 Command1\n"
		"\tPid2 IfExited2 ExitStatus2 Command2\n"
		"\tPid3 IfExited3 ExitStatus3 Command3\n"
		"\t...\n"
		"\n";

	fputs(s, fp);
}

int main(int argc, char *argv[]){
	if(!parseArgs(argc, argv)){
		usage(stderr);
		exit(255);
	}
	runParallel();
}
