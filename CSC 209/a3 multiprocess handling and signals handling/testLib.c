#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include "libParseArgs.h"
#include "libProcessControl.h"

int testCreateCommand(){
	char *s;

       	printf("%s\n",createCommand("that", "THIS"));
       	printf("%s\n",createCommand("{}that", "THIS"));
       	printf("%s\n",createCommand("t{}hat", "THIS"));
       	printf("%s\n",createCommand("th{}at", "THIS"));
       	printf("%s\n",createCommand("tha{}t", "THIS"));
       	printf("%s\n",createCommand("that{}", "THIS"));

       	printf("%s\n",createCommand("{}{}that", "THIS"));
       	printf("%s\n",createCommand("{}t{}hat", "THIS"));
       	printf("%s\n",createCommand("{}th{}at", "THIS"));
       	printf("%s\n",createCommand("{}tha{}t", "THIS"));
       	printf("%s\n",createCommand("{}that{}", "THIS"));

       	printf("%s\n",createCommand("t{}{}hat", "THIS"));
       	printf("%s\n",createCommand("t{}h{}at", "THIS"));
       	printf("%s\n",createCommand("t{}ha{}t", "THIS"));
       	printf("%s\n",createCommand("t{}hat{}", "THIS"));

       	printf("%s\n",createCommand("th{}{}at", "THIS"));
       	printf("%s\n",createCommand("th{}a{}t", "THIS"));
       	printf("%s\n",createCommand("th{}at{}", "THIS"));

       	printf("%s\n",createCommand("th{}{}at", "THIS"));
       	printf("%s\n",createCommand("th{}a{}t", "THIS"));
       	printf("%s\n",createCommand("{}th{}at{}", "THIS"));
       	printf("%s\n",createCommand("th{}{}at{}", "THIS"));
       	printf("%s\n",createCommand("th{}a{}t{}", "THIS"));
       	printf("%s\n",createCommand("th{}at{}{}", "THIS"));
}

int main(int argc, char ** argv){
	int retVal=parseArgs(argc, argv);
	printf("retVal=%d\n", retVal);
	if(retVal)printParallelParams();

	testCreateCommand();
}
