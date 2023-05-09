extern int iWord, iFileName;

/**
 * parse and understand the args. 
 * Returning from this indicates that static variables 
 * iStartWords=-1, iEndWords=-1; int iFileName=-1; are all properly initialized
 * returns 1 for success, 0 for error
 */
int parseArgs(int argc, char *argv[]);
