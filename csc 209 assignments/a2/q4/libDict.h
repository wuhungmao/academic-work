/**
 * A single key, value pair in the dict.
 */
typedef struct {
	char *key;
	void *value; // So we can point to anything
} DictEntry;

/**
 * All elements pointed to in this row hash to the same value
 */
typedef struct {
	int numEntries; // The current number of key, value pairs in this row
	int capacity; // The maximum number of key value pairs in this row
	DictEntry *entries;
} DictRow;

/**
 * A Dictionary consists of a sequence of rows. Each row 
 * has a sequence of (key ,value) pairs, all keys hashed 
 * to index appear in rows[index]
 */
typedef struct {
	int numRows; 
	DictRow *rows;
} Dict;

/**
 * hash *c as a sequence of bytes mod m
 */
int dictHash(char *c, int m);

/**
 * Print the dictionary, 
 * level==0, dict header
 * level==1, dict header, rows headers
 * level==2, dict header, rows headers, and keys
 */
void dictPrint(Dict *d, int level);

/**
 * Return the DictEntry for the given key, NULL if not found.
 * This is so we can store NULL as a value.
 */
DictEntry *dictGet(Dict *d, char *key);

/**
 * Delete key from dict if its found in the dictionary
 * Returns 1 if found and deleted
 * Returns 0 otherwise
 */
int dictDel(Dict *d, char *key);


/**
 * put (key, value) in Dict
 * return 1 for success and 0 for failure
 */
int dictPut(Dict *d, char *key, void *value);

/**
 * free all resources allocated for this Dict. Everything, and only those things
 * allocated by this code should be freed.
 */
void dictFree(Dict *d);

/**
 * Allocate and initialize a new Dict. Initially this dictionary will have initRows
 * hash slots. If initRows==0, then it defaults to DICT_INIT_ROWS
 * Returns the address of the new Dict on success
 * Returns NULL on failure
 */
Dict * dictNew(int initRows);


