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

#define DICT_INIT_ENTRIES 8
#define DICT_GROW_FACTOR 2 

/**
 * Print the dictionary, 
 */
void dictPrint(Dict *d, int level){
	if(d==NULL){
		printf("\tDict==NULL\n");
		return;
	}
	printf("\tDict: numEntries=%d capacity=%d [", d->numEntries, d->capacity);
	if(level>=2){
		for(int j=0;j<d->numEntries;j++){
			printf("(%s,%d), ",d->entries[j].key, d->entries[j].value);
		}
	}
	printf("]\n");
}

/**
 * Return the DictEntry for the given key, NULL if not found.
 */
DictEntry *dictGet(Dict *d, char *key){
	int i = 0;
	while (i < d -> numEntries) {
		if (strcmp((d ->entries)[i].key, key) == 0) {
			return &(d -> entries)[i];
		}
		i++;
	}
	// find key in row
	return NULL;
}

/**
 * Delete key from Dict if its found in the dictionary
 * Returns 1 if found and deleted
 * Returns 0 otherwise
 */
int dictDel(Dict *d, char *key){
	#ifdef DEBUG
	printf("dictDel(d,%s)\n",key);
	dictPrint(d,DEBUG_LEVEL);
	#endif
	int i = 0;
	while (i < d ->numEntries) {
		if (strcmp((d ->entries)[i].key, key) == 0) {
			char *temp = d -> entries[i].key;
			free(temp);
			d ->numEntries--;
			break;
		
		}
		i++;
	}
	if (i == d-> numEntries+1) {
		return 0;
	}
	while (i < d-> numEntries ) {
		((d -> entries)[i].key) = (d ->entries[i+1].key);
		((d -> entries)[i].value) = (d ->entries[i+1].value);
		i++;
	}
	/*	free(d -> entries[i].key);*/
	d ->entries[i].key = NULL;
	d -> entries[i].value = 0;

	// find key 
	// free key
	// Move everything over

	#ifdef DEBUG
	dictPrint(d,DEBUG_LEVEL);
	#endif

	return 1;
}

/**
 * put (key, value) in Dict
 * return 1 for success and 0 for failure
 */
int dictPut(Dict *d, char *key, int value){
	#ifdef DEBUG
	printf("dictPut(d,%s)\n",key);
	dictPrint(d,DEBUG_LEVEL);
	#endif
	if (key == NULL) {
		return 0;
	}
	// If key is already here, just replace value
	for(int i = 0; i< d->numEntries; i++){
		if (strcmp((d -> entries + i)->key, key) == 0) {
			(d -> entries + i)->value = value;
		}
	}

	#ifdef DEBUG
	dictPrint(d,DEBUG_LEVEL);
	#endif

	/** 
	 * At this point we know the key is not in Dict, so we 
	 * need to place (key, value) as a new entry in this 
	 *
	 * if there is no space, expand the row
	 */
	if (d -> capacity == d -> numEntries) {
		d -> capacity *= DICT_GROW_FACTOR;
		d -> entries = realloc(d ->entries, sizeof(DictEntry)*d->capacity);
		if (d->entries == NULL){
			return 0;
		}
		/*after I reallocate, how does new memory know what data type it wants? Does it assume we will add object
		of such data type to the new memory block?*/
		DictEntry entry;
		entry.key= strdup(key);
		entry.value = value;
		*(d -> entries + (d ->numEntries)) = entry;
		d -> numEntries++;
		#ifdef DEBUG
		dictPrint(d,DEBUG_LEVEL);
		#endif
		return 1;
	} else if (d -> capacity > d -> numEntries) {
	/**
	 * We now know we have space.
	 * This is a new key for this row, so we want to place (key, value)
	 *
	 * In python only immutables can be hash keys. If the user can change the key sitting
	 * in the Dict, then we won't be able to find it again. We solve this problem here
	 * by copying keys using strdup.
	 * 
	 */
		(d -> entries)[d -> numEntries].key=strdup(key);
		(d -> entries)[d -> numEntries].value=value;
		(d -> numEntries)++;
		#ifdef DEBUG
		dictPrint(d,DEBUG_LEVEL);
		#endif
		return 1;}
}

/**
 * free all resources allocated for this Dict. Everything, and only those things
 * allocated by this code should be freed.
 */
void dictFree(Dict *d){
	for(int i = 0; i< d->numEntries+1; i++){
		free((d -> entries + i)->key);
	}
	for(int i = 0; i< d->numEntries+1; i++){
		free(d -> entries + i);
	}
	free(d);
	// free all the keys we allocated
	// free the array of DictEntry
	// free Dict
}

/**
 * Allocate and initialize a new Dict. 
 * Returns the address of the new Dict on success
 * Returns NULL on failure
 */
Dict * dictNew(){
	Dict *d = NULL;

	// Create the Dict and initialize it
	d = (Dict *) malloc(sizeof(Dict));
	d -> capacity = DICT_INIT_ENTRIES;
	d -> numEntries = 0;

	// Create the DictEntry array 
	d -> entries = NULL;
	if ((d -> entries = malloc(sizeof(DictEntry) * DICT_INIT_ENTRIES)) == NULL) {
		free(d);
		return NULL;
	}
	d -> entries->key = NULL;

	for (int i = 0; i<DICT_INIT_ENTRIES; i++) {
		dictPut(d, NULL, -1);
	}

	/**
	 * Initialize all of the entries to (NULL, -1). Assuming the rest of our code
	 * is OK, we really should not have to do this. But lets do it
	 * anyway. In this case, entries has 
	 *
	 *     numEntries = 0
	 *     capacity   = DICT_INIT_ENTRIES
	 */
	return d;
}

