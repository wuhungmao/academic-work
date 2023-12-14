/**
 * create and return a newly malloced command from commandTemplate and argument
 * the new command replaces each occurrance of {} in commandTemplate with argument
 */
char *createCommand(char *commandTemplate, char *argument);
int runParallel();

