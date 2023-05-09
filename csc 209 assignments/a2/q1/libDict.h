/**
 * A single key, value pair in the dict.
 */
typedef struct {
	char *key;
	int value; 
} DictEntry;

/**
 * A list of key, value pairs
 */
typedef struct {
	int numEntries; // The current number of key, value pairs in this row
	int capacity; // The maximum number of key value pairs in this row
	DictEntry *entries;
} Dict;

/**
 * Print the dictionary, 
 */
void dictPrint(Dict *d, int level);

/**
 * Return the DictEntry for the given key, NULL if not found.
 * This is so we can store NULL as a value.
 */
DictEntry *dictGet(Dict *d, char *key);

/**
 * Delete key from Dict if its found in the dictionary
 * Returns 1 if found and deleted
 * Returns 0 otherwise
 */
int dictDel(Dict *d, char *key);


/**
 * put (key, value) in Dict
 * return 1 for success and 0 for failure
 */
int dictPut(Dict *d, char *key, int value);

/**
 * free all resources allocated for this Dict. Everything, and only those things
 * allocated by this code should be freed.
 */
void dictFree(Dict *d);

/**
 * Allocate and initialize a new Dict. 
 * Returns the address of the new Dict on success
 * Returns NULL on failure
 */
Dict * dictNew();


