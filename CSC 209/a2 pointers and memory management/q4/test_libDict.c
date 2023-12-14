#include <stdio.h>
#include <stdlib.h>
#include<string.h>
#include "libDict.h"
#define BUFF_LEN 1024

int test_hash(){
	printf("BEGIN: %s\n", __FUNCTION__);
	char buff[BUFF_LEN];
	int numKeys = 1<<20;
	int first=1;
	for(int m=1<<8;m<1<<15;m=m<<1){
		int h[m]; // to keep collisions
		for(int i=0;i<m;i++)h[i]=0; // clear collisions
		for(int i=0;i<numKeys;i++){
			sprintf(buff, "key %d", i);
			// printf("hash(%s,%d)=%d\n", buff,m,dictHash(buff,m));
			h[dictHash(buff, m)]++;
		}
		/**
		 * Lets say hash is good if the distribution essentially
		 * looks uniform. All entries should be within (numKeys/m)/2 and (numKeys/m)*2
		 */
		int collision_min = (numKeys/m)/2, collision_max = (numKeys/m)*2;
		int numBadRows=0;
		for(int i=0;i<m;i++){
			if(! (collision_min<h[i] && h[i]<collision_max) ){
				numBadRows++;
			}
		}
		printf("HASH: numBadRows/numRows = %d/%d\n", numBadRows, m);
		/**
		if(first){
			first = !first;
			for(int i=0;i<m;i++)printf("h[%d]=%d\n", i,h[i]);
		}
		**/

	}
	printf("END  : %s\n", __FUNCTION__);
}

int test_dictNew(){
	printf("BEGIN: %s\n", __FUNCTION__);

	Dict *d=dictNew(10);
	dictPrint(d,2);
	dictFree(d);

	printf("END  : %s\n", __FUNCTION__);
}

int test_dictGet_empty(){
	printf("BEGIN: %s\n", __FUNCTION__);

	Dict *d=dictNew(10);
	char *keys[]={ "", "a", "aa", "ab", "this", "that", "this is a very long string with lots of characters"};
	int keysLen = sizeof(keys)/sizeof(keys[0]); // aka sizeof(keys)/sizeof(char *);

	for(int i=0;i<keysLen;i++){
		DictEntry *de = dictGet(d,keys[i]);
		if(de==NULL){
			// printf("OK   : searched for %s on empty Dict and result was NULL\n", keys[i]);
		}else{
			printf("ERROR: searched for %s on empty Dict and result was NOT NULL\n", keys[i]);
		}
	}
	dictFree(d);

	printf("END  : %s\n", __FUNCTION__);
}

/**
 * Check that words at index%m==t are exactly the words
 * in the dictionary
 */
int checkWords(Dict *d, FILE *fp, int numWords, int m, int t){
	printf("BEGIN: %s\n", __FUNCTION__);

	int numWordsProcessed = 0;
	char buff[BUFF_LEN];

	rewind(fp);
	while(fgets(buff, BUFF_LEN, fp)!=NULL){
		buff[strcspn(buff, "\r\n")] = '\0'; // remove trailing newline
		DictEntry *de=dictGet(d,buff);
		if(numWordsProcessed%m==t){
			if(de==NULL){
				printf("ERROR: Key %s not found\n", buff);
				continue;
			}
			char *key=de->key, *value=de->value;
			if(strcmp(buff, de->value)!=0){
				printf("ERROR: %s found, but value incorrect\n", buff);
				continue;
			}
		} else {
			if(de!=NULL){
				printf("ERROR: Deleted key %s still in Dict\n", buff);
				continue;
			}
		}
		numWordsProcessed++;
		if(numWordsProcessed==numWords)break;
	}
	printf("END  : %s\n", __FUNCTION__);
}

/**
 * Add words at index%m==t to the dictionary
 */
int addWords(Dict *d, FILE *fp, int numWords, int m, int t){
	printf("BEGIN: %s\n", __FUNCTION__);
	int numWordsProcessed = 0;
	char buff[BUFF_LEN];
	rewind(fp);

	numWordsProcessed = 0;
	while(fgets(buff, BUFF_LEN, fp)!=NULL){
		buff[strcspn(buff, "\r\n")] = '\0'; // remove trailing newline
		if(numWordsProcessed%m==t){
			dictPut(d,buff,strdup(buff));
		}
		numWordsProcessed++;
		if(numWordsProcessed==numWords)break;
	}
	printf("END  : %s\n", __FUNCTION__);
}

/**
 * Delete words at index%m==t from the dictionary
 */
int delWords(Dict *d, FILE *fp, int numWords, int m, int t){
	printf("BEGIN: %s\n", __FUNCTION__);
	int numWordsProcessed = 0;
	char buff[BUFF_LEN];

	rewind(fp);
	while(fgets(buff, BUFF_LEN, fp)!=NULL){
		buff[strcspn(buff, "\r\n")] = '\0'; // remove trailing newline
		if(numWordsProcessed%m==t){
			DictEntry *de=dictGet(d,buff);
			if(de!=NULL){
				char *key=de->key, *value=de->value;
				if(!dictDel(d,buff))printf("ERROR (delete): Can't delete %s\n", buff);
				free(value);
			}
		}
		numWordsProcessed++;
		if(numWordsProcessed==numWords)break;
	}
	printf("END  : %s\n", __FUNCTION__);
}

int test_stressTest(int numRows, int numWords){
	printf("BEGIN: %s\n", __FUNCTION__);

	int numWordsProcessed = 0;
	char buff[BUFF_LEN];
	Dict *d=dictNew(numRows);

	char *fileName = "/usr/share/dict/words";
	FILE *fp = fopen(fileName,"r");
	if(fp==NULL){
		perror(fileName);
		exit(1);
	}


	addWords(d,fp,numWords,1,0);
	checkWords(d,fp,numWords,1,0);
	delWords(d,fp,numWords,2,1);
	checkWords(d,fp,numWords,2,0);
	delWords(d,fp,numWords,2,0);
	checkWords(d,fp,numWords,2,3); // No words should be present

	fclose(fp);

	dictFree(d);
	printf("END  : %s\n", __FUNCTION__);
}

int main(int argc, char **argv){
	test_dictNew();
	/*
	test_hash();*/
	test_dictGet_empty();
	test_stressTest(11,150);
}
