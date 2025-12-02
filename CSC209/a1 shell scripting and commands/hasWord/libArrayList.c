#include <stdlib.h>
#include "libArrayList.h"
#define AL_GROW_SIZE 10
/**
 * Return value of integer at position index, if 0<=index<a->size
 * Return 0, otherwise
 */
int alGet(ArrayList *a, int index){
	if(0<=index && index<a->size)return a->elements[index];
	else return 0;
}

/**
 * put value at position index if 0<=index<a->size
 * return 1 for success and 0 for failute
 */
int alSet(ArrayList *a, int index, int value){
	if(0<=index && index<a->size){
		*((a->elements)+index) = value;
		return 1;
	} else return 0;
}

/**
 * put value as the new last value in the ArrayList
 * growing the ArrayList if necessary
 * Return 1 if success, 0 otherwise
 */
int alAdd(ArrayList *a, int value){
	if(a->size>=a->capacity){
		// Need to realloc a->elements, increase the capacity by AL_GROW_SIZE
		a->capacity=a->capacity+AL_GROW_SIZE;
		a->elements=(int *)realloc(a->elements, sizeof(int)*a->capacity);
		int *newElements;
	}
	a->size++;
        alSet(a, a->size-1, value);
        return 1;
}

/**
 * Allocate and initialize a new ArrayList
 * Returns the address of the new ArrayList if everything is OK
 * Returns NULL if anything goes wrong
 */
ArrayList * alNew(){
	ArrayList *a;
	// Allocate a new ArrayList
	if((a=malloc(sizeof(ArrayList))) == NULL)return NULL;
		// set its size to 0
	a->size=0;
	// set its capacity to AL_GROW_SIZE
	a->capacity=AL_GROW_SIZE;
		// allocate capacity integers and point a->elements to them
	if((a->elements=malloc(a->capacity*sizeof(int)))==NULL) {
		free(a);
		return NULL;
	}
	return a;
}

