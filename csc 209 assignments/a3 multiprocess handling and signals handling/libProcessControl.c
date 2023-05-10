#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<fcntl.h>
#include<unistd.h>
#include<signal.h>
#include<sys/types.h>
#include<sys/wait.h>
#include<sys/stat.h>

#include "libParseArgs.h"
#include "libProcessControl.h"

/**
 * parallelDo -n NUM -o OUTPUT_DIR COMMAND_TEMPLATE ::: [ ARGUMENT_LIST ...]
 * build and execute shell command lines in parallel
 */

/**
 * create and return a newly malloced command
 * from commandTemplate and argument the new command replaces each occurrance of {} in commandTemplate with argument
 */
char *createCommand(char *commandTemplate, char *argument){
	char *command=NULL;
	int total_len = strlen(commandTemplate) + strlen(argument);
	command = malloc(sizeof(char) * total_len);
	int k = 0;
	char left_brac = '{';
	char right_brac = '}';
	for (int i = 0; i<strlen(commandTemplate); i++) {
		if (commandTemplate[i] == left_brac) {
			for (int j = 0; j<strlen(argument); j++) {
				command[k] = argument[j];
				k++;
			}
			continue;
		} else if (commandTemplate[i] == right_brac) {
			continue;
		}
		command[k] = commandTemplate[i];
		k++;
	}
	return command;
}

typedef struct PROCESS_STRUCT {
	int pid;
	int ifExited;
	int exitStatus;
	int status;
	char *command;
} PROCESS_STRUCT;

typedef struct PROCESS_CONTROL {
	int numProcesses;
	int numRunning; 
	int maxNumRunning;
	int numCompleted;
	PROCESS_STRUCT *process;
} PROCESS_CONTROL;

PROCESS_CONTROL processControl;

void printSummary(){
	printf("%d %d %d\n", processControl.numProcesses, processControl.numCompleted, processControl.numRunning);
}

void printSummaryFull(){
    printSummary();
    int i=0, numPrinted=0;
    while(numPrinted<processControl.numCompleted && i<processControl.numProcesses){
			if(processControl.process[i].ifExited){
					printf("%d %d %d %s\n",
							processControl.process[i].pid,
							processControl.process[i].ifExited,
							processControl.process[i].exitStatus,
							processControl.process[i].command);
					numPrinted++;
			}
			i++;
	}
}

/**
 * find the record for pid and update it based on status
 * status has information encoded in it, you will have to extract it
 * 
 * what should child status be before it terminate?
 */

void updateStatus(int pid, int status){
	for (int i = 0; i<processControl.numProcesses; i++) {
		if (processControl.process[i].pid == pid) {
			processControl.process[i].status = status;
			if (WIFEXITED(status)) {
				processControl.process[i].exitStatus = WEXITSTATUS(status);
			}
			processControl.process[i].ifExited = 1;
		}
	}
}

void handler(int signum){
	if (signum == SIGUSR1) {
		printf("%d %d %d\n", processControl.numProcesses, processControl.numCompleted, processControl.numRunning);
	} else if (signum == SIGUSR2) {
		printf("%d %d %d\n", processControl.numProcesses, processControl.numCompleted, processControl.numRunning);
		for (int i = 0; i<processControl.numCompleted;i++){
			printf("%d %d %d %s\n", processControl.process[i].pid, processControl.process[i].ifExited, processControl.process[i].exitStatus, processControl.process[i].command);
		}
	}
}

/**
 * This function does the bulk of the work for parallelDo. This is called
 * after understanding the command line arguments. runParallel 
 * uses pparams to generate the commands (createCommand), 
 * forking, redirecting stdout and stderr, waiting for children, ...
 * fprintf(stdout)
 * errno, strerror() fprinf, perror(), sprintf()
 * Instead of passing around variables, we make use of globals pparams and
 * processControl. 
  
 
 Are we required to use sigaction() with struct sigaction or signal()?
 Does it matter where we put signal() or sigaction() in our code, they will be triggered once a process received a signal? 
 */


int runParallel(){
	/*Associate SIGUSR1 and SIGUSR2 with signal handling*/

	signal(SIGUSR1, handler);
	signal(SIGUSR2, handler);

	/*provide information for processControl*/
	processControl.numProcesses = pparams.argumentListLen;
	processControl.numRunning=0;
	processControl.numCompleted=0;
	processControl.maxNumRunning = pparams.maxNumRunning;
	processControl.process = (PROCESS_STRUCT *) malloc(sizeof(PROCESS_STRUCT) * processControl.numProcesses);

	/*mkdir make directory*/
	if (mkdir(pparams.outputDir, 0777)==-1) {
				perror("mkdir fail");
	}

	/*Start processing commands*/
	for (int i = 0; i<processControl.numProcesses; i++) {
		char *command;
		command = createCommand(pparams.commandTemplate, pparams.argumentList[i]);
		int pid = fork();
		processControl.numRunning++;

		/*process information in each child process*/
		processControl.process[i].command = command;
		processControl.process[i].pid = pid;
		processControl.process[i].ifExited = 0;

		if (pid == 0) {
			/*redirect stdout and stderr to outputDir*/
			char *outfile = malloc(sizeof(char) * 100);
			char *errfile = malloc(sizeof(char) * 100);
			sprintf(outfile, "%s/%d.stdout", pparams.outputDir, getpid());
			sprintf(errfile, "%s/%d.stderr", pparams.outputDir, getpid());
			/*
			printf("%s is outfile\n", outfile);
			printf("%s is errfile\n", errfile);
			*/
			int outfd;
			if ((outfd = open(outfile, O_CREAT|O_RDWR, 0666)) == -1){
				printf("outfd:%d\n", outfd);
			}
			
			int errfd;
			if ((errfd = open(errfile, O_CREAT|O_RDWR, 0666)) == -1){
				printf("errfd:%d\n", errfd);
			}

			printf("outfd:%d", outfd);
			if (dup2(outfd, 1) == -1){
				perror("dup2 failed");
			}

			if (dup2(errfd, 2) == -1) {
				perror("dup2 failed");
			}

			close(outfd);
			close(errfd);

			/*
			command = echo 1, ./isPrime 1000000000, ...   
			process terminates and sends result to terminal before c could redirect standard output to a file
			*/
			/* https://piazza.com/class/lcnmbsmgyth7fa/post/1125*/
			
			execl("/bin/bash", "/bin/bash", "-c", command, (char *) NULL); 
			exit(1);
		}

		if (processControl.numRunning == processControl.maxNumRunning) {
			int status;
			int finished_child_pid=wait(&status);
			updateStatus(finished_child_pid, status);
			processControl.numRunning--;
			processControl.numCompleted++;
		}
	}

	int remain_process = processControl.numRunning;
	for (int i = 0; i<remain_process; i++) {
		/*retrieve status of remaining running processes*/
		int status;
		int finished_child_pid=wait(&status);
		updateStatus(finished_child_pid, status);
		processControl.numRunning--;
		processControl.numCompleted++;
	}
	// YOUR CODE GOES HERE
	// THERE IS A LOT TO DO HERE!!
	// TAKE SMALL STEPS, MAKE SURE THINGS WORK AND THEN MOVE FORWARD.
	printSummaryFull();
}
