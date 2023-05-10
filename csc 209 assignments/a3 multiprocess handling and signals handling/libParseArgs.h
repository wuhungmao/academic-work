typedef struct {
	int maxNumRunning;
	char *outputDir;
	char *commandTemplate;
	char **argumentList; 
	int argumentListLen;
} PARALLEL_PARAMS;

extern PARALLEL_PARAMS pparams;

void printParallelParams();

int parseArgs(int argc, char *argv[]);
