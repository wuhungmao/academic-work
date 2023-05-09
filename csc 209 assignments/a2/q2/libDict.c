#include<stdio.h> 
#include <stdlib.h>
#include <string.h>
#include "libDict.h"

#define DEBUG
#define DEBUG_LEVEL 3

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

#define DICT_INIT_ROWS 1024 
#define DICT_GROW_FACTOR 2
#define ROW_INIT_ENTRIES 8
#define ROW_GROW_FACTOR 2 

#define PRIME1 77933 // a large prime
#define PRIME2 119557 // a large prime

/**
 * hash *c as a sequence of bytes mod m
 */
int dictHash(char *c, int m){
	int sum=0;
	while(*c!='\0'){
		int num = *c; 
		sum+= PRIME1*num+PRIME2*sum;
		c++;
	}
	if(sum<0)sum=-sum;
	sum = sum%m;
	return sum;
}

/**
 * Print the dictionary, 
 * level==0, dict header
 * level==1, dict header, rows headers
 * level==2, dict header, rows headers, and keys
 */
void dictPrint(Dict *d, int level){
	if(d==NULL){
		printf("\tDict==NULL\n");
		return;
	}
	printf("Dict\n");
	printf("\tnumRows=%d\n",d->numRows);
	if(level<1)return;

	for(int i=0;i<d->numRows;i++){
		printf("\tDictRow[%d]: numEntries=%d capacity=%d keys=[", i, d->rows[i].numEntries, d->rows[i].capacity);
		if(level>=2){
			for(int j=0;j<d->rows[i].numEntries;j++){
				printf("%s, ",d->rows[i].entries[j].key);
			}
		}
		printf("]\n");
	}
}

/**
 * Return the DictEntry for the given key, NULL if not found.
 * This is so we can store NULL as a value.
 */
DictEntry *dictGet(Dict *d, char *key){
	// find row
	int h = dictHash(key, d->numRows);

	// find key in row
	for (int i = 0; i<((d->rows) + h)->numEntries; i++) {
		if (strcmp((((d->rows) + h) ->entries + i) ->key, key) == 0) {
			return (((d->rows) + h) ->entries + i);
		}
	}
	return NULL;
}

/**
 * Delete key from dict if its found in the dictionary
 * Returns 1 if found and deleted
 * Returns 0 otherwise
 */
int dictDel(Dict *d, char *key){
	// find row
	int h = dictHash(key, d->numRows);
	
	#ifdef DEBUG
	printf("dictDel(d,%s) hash=%d\n",key, h);
	dictPrint(d,DEBUG_LEVEL);
	#endif

	if(((d -> rows) + h) -> numEntries == 1){
	
		free((((d->rows) + h) ->entries + 0) -> key);
		(((d->rows) + h) ->entries + 0) -> value = 0;
		((d -> rows) + h) -> numEntries--;
		return 1;
	}
	
	// find key in row
	int found = 0;
	for (int i = 0; i<((d->rows) + h)->numEntries-1; i++) {
		if (found == 1) {
			free((((d->rows) + h) ->entries + i) ->key);
			(((d->rows) + h) ->entries + i) -> key = strdup((((d->rows) + h) ->entries + i+1) ->key);
			(((d->rows) + h) ->entries + i) -> value = (((d->rows) + h) ->entries + i+1) -> value;
		}
		if (strcmp((((d->rows) + h) ->entries + i) ->key, key) == 0) {
			// free key

			free((((d->rows) + h) ->entries + i) ->key);
			found = 1;
			(((d->rows) + h) ->entries + i) -> key = strdup((((d->rows) + h) ->entries + i+1) ->key);
			(((d->rows) + h) ->entries + i) -> value = (((d->rows) + h) ->entries + i+1) -> value;
			char * thekey= (((d->rows) + h) ->entries + i) -> key;
		}
	}

	((d -> rows) + h) -> numEntries--;
	if (strcmp((((d->rows) + h) ->entries + ((d->rows) + h)->numEntries) ->key, key) == 0) {
		free((((d->rows) + h) ->entries + ((d->rows) + h)->numEntries) ->key);
			return 1;
	}
	if (found == 0) {
		return 0;
	}

	int numentries = ((d -> rows) + h) -> numEntries;

	free(((d->rows + h) ->entries + numentries) -> key);
	(((d->rows) + h) ->entries + ((d -> rows) + h) -> numEntries) -> value = 0;
	// Move everything over


	#ifdef DEBUG
	dictPrint(d,DEBUG_LEVEL);
	#endif

	return 1;
}

/**
 * put (key, value) in Dict
 * return 1 for success and 0 for failure
 *
* This is a new key for this row, so we want to place the key, value pair
* In python only immutables can be hash keys. If the user can change the key sitting
* in the Dict, then we won't be able to find it again. We solve this problem here
* by copying keys using strdup.
* 
* At this point we know there is space, so copy the key and place it in the row
* along with its value.
*
/*
 */
int dictPut(Dict *d, char *key, void *value){
	int h = dictHash(key, d->numRows);
	#ifdef DEBUG
	printf("dictPut(d,%s) hash=%d\n",key, h);
	dictPrint(d,DEBUG_LEVEL);
	#endif
	// If key is already here, just replace value
	if (((d ->rows) + h) ->numEntries != 0) {
		for (int i = 0; i<((d ->rows) + h) ->numEntries; i++) {
			if (strcmp((((d ->rows) + h) ->entries + i) ->key, key) == 0) {
				(((d ->rows) + h) ->entries + i) -> value = value;
			}
		}
	}
	
	#ifdef DEBUG
	dictPrint(d,DEBUG_LEVEL);
	#endif
	/*
	 * else we need to place (key,value) as a new entry in this row
	 * if there is no space, expand the row
	 */
	if (((d ->rows) + h) ->numEntries == ((d ->rows) + h) ->capacity) {
		int place_for_new = ((d -> rows) + h) ->numEntries;
		((d -> rows) + h) -> capacity = ((d -> rows) + h) ->capacity*DICT_GROW_FACTOR;
		if ((((d -> rows) + h) -> entries = (DictEntry *) realloc(((d -> rows) + h) -> entries, ((d -> rows) + h) ->capacity * sizeof(DictEntry))) == NULL) {
			free(((d -> rows) + h) -> entries);
			return 0;
		};
		(((d -> rows) + h) -> entries + place_for_new) -> key = strdup(key);
		(((d -> rows) + h) -> entries + place_for_new) -> value = value;
		((d -> rows) + h) ->numEntries++;
	} else if (((d ->rows) + h) ->numEntries < ((d ->rows) + h) ->capacity) {
		
		#ifdef DEBUG
		dictPrint(d,DEBUG_LEVEL);
		#endif
		int place_for_new = ((d -> rows) + h) ->numEntries;
		(((d -> rows) + h) -> entries + place_for_new) -> key = strdup(key);
		char *to_be_free = (((d -> rows) + h) -> entries + place_for_new) -> key;
		(((d -> rows) + h) -> entries + place_for_new) -> value = value;
		((d -> rows) + h) ->numEntries++;
	}
	
	#ifdef DEBUG
	dictPrint(d,DEBUG_LEVEL);
	#endif
	return 1;
}

/**
 * free all resources allocated for this Dict. Everything, and only those things
 * allocated by this code should be freed.
 */
void dictFree(Dict *d){
	for (int i = 0; i<d ->numRows; i++) {

		for (int j = 0; j < ((d->rows) + i)->numEntries; j++) {

			free((((d -> rows) + i)->entries+j)->key);
			(((d -> rows) + i)->entries+j)->key = NULL;

			free((((d -> rows) + i)->entries+j)->value);
		}

		free((d->rows+i) ->entries);
		(d->rows+i) ->entries = NULL;
	}

	free(d->rows);
	d->rows = NULL;
	d ->numRows=0;
	free(d);
}

/**
 * Allocate and initialize a new Dict. Initially this dictionary will have initRows
 * hash slots. If initRows==0, then it defaults to DICT_INIT_ROWS
 * Returns the address of the new Dict on success
 * Returns NULL on failure
 */
Dict * dictNew(int initRows){
	Dict *d=NULL;
	if ((d = malloc(sizeof(Dict))) == NULL) {
		free(d);
		return NULL;
	};
	if (initRows == 0){

		d -> numRows = DICT_INIT_ROWS;
		if ((d -> rows = (DictRow *) malloc(sizeof(DictRow) * DICT_INIT_ROWS)) == NULL) {
			free(d -> rows);
			return NULL;
		}

		for (int i = 0; i <DICT_INIT_ROWS; i++){
			((d -> rows) + i)->capacity = ROW_INIT_ENTRIES;
			if ((((d -> rows) + i)->entries = (DictEntry *) malloc(sizeof(DictEntry) * ROW_INIT_ENTRIES)) == NULL) {
				free(((d -> rows) + i)->entries);
				return NULL;
			} 
			
			for (int j = 0; j < ROW_INIT_ENTRIES; j++) {
				((((d -> rows) + i) -> entries) + j) -> key = NULL;
			((d -> rows) + i) -> numEntries = 0;
			}
		}
	}

	else if (initRows != 0) {

		d -> numRows = initRows;
		if ((d -> rows = (DictRow *) malloc(sizeof(DictRow) * initRows)) == NULL) {

			free(d);
			return NULL;
		}
		
		for (int i = 0; i <initRows; i++){
			((d -> rows) + i)->capacity = ROW_INIT_ENTRIES;
			if ((((d -> rows) + i)->entries = (DictEntry *) malloc(sizeof(DictEntry) * ROW_INIT_ENTRIES)) == NULL) {
				
				free(((d -> rows) + i)->entries);
				return NULL;
			}
			
			for (int j = 0; j < ROW_INIT_ENTRIES; j++) {
				((((d -> rows) + i) -> entries) + j) -> key = NULL;
			((d -> rows) + i) -> numEntries = 0;
			}
		}
	}
	return d;
}

