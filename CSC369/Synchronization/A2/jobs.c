
// ------------
// This code is provided solely for the personal and private use of
// students taking the CSC369H5 course at the University of Toronto.
// Copying for purposes other than this use is expressly prohibited.
// All forms of distribution of this code, whether as given or with
// any changes, are expressly prohibited.
//
// Authors: Bogdan Simion
//
// All of the files in this directory and all subdirectories are:
// Copyright (c) 2025 Bogdan Simion
// -------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include "executor.h"
extern struct executor tassadar;

/**
 * Populate the job lists by parsing a file where each line has
 * the following structure:
 *
 * <id> <type> <num_resources> <resource_id_0> <resource_id_1> ...
 *
 * Each job is added to the queue that corresponds with its job type.
 */
void parse_jobs(char *file_name)
{
    int id;
    struct job *cur_job;
    struct admission_queue *cur_queue;
    enum job_type jtype;
    int num_resources, i;
    FILE *f = fopen(file_name, "r");
    /* parse file */
    while (fscanf(f, "%d %d %d", &id, (int *)&jtype, (int *)&num_resources) == 3)
    {
        /* construct job */
        cur_job = malloc(sizeof(struct job));
        cur_job->id = id;
        cur_job->type = jtype;
        cur_job->num_resources = num_resources;
        cur_job->resources = malloc(num_resources * sizeof(int));
        int resource_id;
        for (i = 0; i < num_resources; i++)
        {
            fscanf(f, "%d ", &resource_id);
            cur_job->resources[i] = resource_id;
            tassadar.resource_utilization_check[resource_id]++;
        }
        assign_processor(cur_job);
        /* append new job to head of corresponding list */
        cur_queue = &tassadar.admission_queues[jtype];
        cur_job->next = cur_queue->pending_jobs;
        cur_queue->pending_jobs = cur_job;
        cur_queue->pending_admission++;
    }
    fclose(f);
}

/*
 * Magic algorithm to assign a processor to a job.
 */
void assign_processor(struct job *job)
{
    int i, proc = job->resources[0];
    for (i = 1; i < job->num_resources; i++)
    {
        if (proc < job->resources[i])
        {
            proc = job->resources[i];
        }
    }
    job->processor = proc % NUM_PROCESSORS;
}

void do_stuff(struct job *job)
{
    /* Job prints its id, its type, and its assigned processor */
    printf("%d %d %d\n", job->id, job->type, job->processor);
}

/**
 * TODO: Fill in this function
 *
 * Do all of the work required to prepare the executor
 * before any jobs start coming
 *
 */
void init_executor()
{
    for (int queue_idx = 0; queue_idx < NUM_QUEUES; queue_idx++)
    {
        // initialize mutex and condition variable for each queue
        pthread_mutex_init(&tassadar.admission_queues[queue_idx].lock, NULL);
        pthread_cond_init(&tassadar.admission_queues[queue_idx].admission_cv, NULL);
        pthread_cond_init(&tassadar.admission_queues[queue_idx].execution_cv, NULL);

        // no pending jobs at initalization
        tassadar.admission_queues[queue_idx].pending_jobs = NULL;
        tassadar.admission_queues[queue_idx].pending_admission = 0;

        // at most 10 jobs in a queue
        tassadar.admission_queues[queue_idx].capacity = QUEUE_LENGTH;
        tassadar.admission_queues[queue_idx].num_admitted = 0;
        tassadar.admission_queues[queue_idx].head = 0;
        tassadar.admission_queues[queue_idx].tail = 0;

        // allocate 10 jobs space
        tassadar.admission_queues[queue_idx].admitted_jobs = malloc(QUEUE_LENGTH * sizeof(struct job *));
    }

    for (int processor_idx = 0; processor_idx < NUM_PROCESSORS; processor_idx++)
    {
        // initialize processor mutex
        pthread_mutex_init(&tassadar.processor_records[processor_idx].lock, NULL);
        tassadar.processor_records[processor_idx].num_completed = 0;
        tassadar.processor_records[processor_idx].completed_jobs = NULL;
    }

    for (int resource_idx = 0; resource_idx < NUM_RESOURCES; resource_idx++)
    {
        // initialize executor resource locks
        pthread_mutex_init(&tassadar.resource_locks[resource_idx], NULL);
        tassadar.resource_utilization_check[resource_idx] = 0;
    }
}

/**
 * Helper function for qsort
 */
int compare_resources(const void *resource1_ptr, const void *resource2_ptr)
{
    int resource1 = *(int *)resource1_ptr;
    int resource2 = *(int *)resource2_ptr;

    if (resource1 < resource2)
    {
        return -1;
    }
    else if (resource1 == resource2)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

/**
 * Acquire all resource locks for a job being executed
 */
void acquire_resource_lck(struct job *job_executing, int *resource_lst, int resource_cnt)
{
    for (int resource_idx = 0; resource_idx < resource_cnt; resource_idx++)
    {
        resource_lst[resource_idx] = job_executing->resources[resource_idx];
    }

    // sort resource_lst
    qsort(resource_lst, resource_cnt, sizeof(int), compare_resources);

    // acquire resource lock in increasing order
    for (int resource_idx = 0; resource_idx < resource_cnt; resource_idx++)
    {
        pthread_mutex_lock(&tassadar.resource_locks[resource_lst[resource_idx]]);
        tassadar.resource_utilization_check[resource_lst[resource_idx]]--;
    }
}

/**
 * Release all resource locks for a job
 */
void release_resource_lck(struct job *job_executing, int *resource_lst, int resource_cnt)
{
    for (int resource_idx = 0; resource_idx < resource_cnt; resource_idx++)
    {
        pthread_mutex_unlock(&tassadar.resource_locks[resource_lst[resource_idx]]);
    }
}

/**
 * Execute job
 */
void execute_single_job(struct job *job_executing)
{
    int resource_cnt = job_executing->num_resources;
    int resource_lst[resource_cnt];
    acquire_resource_lck(job_executing, resource_lst, resource_cnt);
    do_stuff(job_executing);
    release_resource_lck(job_executing, resource_lst, resource_cnt);
}

/**
 * Record job finished in processor_record
 */
void record(struct job *job_executing)
{
    struct processor_record *processor_record = &tassadar.processor_records[job_executing->processor];
    // acquire processor_record lock
    pthread_mutex_lock(&processor_record->lock);
    int completed_job = processor_record->num_completed;

    if (completed_job == 0)
    {
        // first completed job
        processor_record->completed_jobs = job_executing;
    }
    else
    {
        // Set most recently finished job as head of the completed jobs
        job_executing->next = processor_record->completed_jobs;
        processor_record->completed_jobs = job_executing;
    }

    processor_record->num_completed++;
    pthread_mutex_unlock(&processor_record->lock);
}

/**
 * TODO: Fill in this function
 *
 * Handles an admission queue passed in through the arg (see the executor.c file).
 * Bring jobs into this admission queue as room becomes available in it.
 * As new jobs are added to this admission queue (and are therefore ready to be taken
 * for execution), the corresponding execute thread must become aware of this.
 *
 */
void *admit_jobs(void *arg)
{
    struct admission_queue *q = arg;
    while (q->pending_admission > 0)
    {
        // obtain queue lock
        pthread_mutex_lock(&q->lock);
        while (q->num_admitted == q->capacity)
        {
            pthread_cond_wait(&q->admission_cv, &q->lock);
        }

        // retrieve first job from pending_jobs list
        struct job *first_job = q->pending_jobs;

        // move to next pending job
        q->pending_jobs = q->pending_jobs->next;

        // break connection between first job to the rest of pending jobs
        first_job->next = NULL;

        // Put first pending job into the queue
        q->admitted_jobs[q->head] = first_job;

        // wrap around if neccessary
        q->head = (q->head + 1) % q->capacity;
        q->num_admitted++;
        q->pending_admission--;
        pthread_cond_signal(&q->execution_cv);
        pthread_mutex_unlock(&q->lock);
    }
    return NULL;
}

/**
 * TODO: Fill in this function
 *
 * Moves jobs from a single admission queue of the executor.
 * Jobs must acquire the required resource locks before being able to execute.
 *
 * Note: You do not need to spawn any new threads in here to simulate the processors.
 * When a job acquires all its required resources, it will execute do_stuff.
 * When do_stuff is finished, the job is considered to have completed.
 *
 * Once a job has completed, the admission thread must be notified since room
 * just became available in the queue. Be careful to record the job's completion
 * on its assigned processor and keep track of resources utilized.
 *
 * Note: No printf statements are allowed in your final jobs.c code,
 * other than the one from do_stuff!
 */
void *execute_jobs(void *arg)
{
    struct admission_queue *q = arg;
    while (1)
    {
        // obtain queue lock
        pthread_mutex_lock(&q->lock);
        if (q->pending_admission == 0 && q->num_admitted == 0)
        {
            // no more job to execute in both the queue and pending job list. Execution thread is done.
            pthread_mutex_unlock(&q->lock);
            break;
        }
        while (q->num_admitted == 0)
        {
            // Execution thread put to sleep because queue is empty
            pthread_cond_wait(&q->execution_cv, &q->lock);
        }
        // Retrieve one job from the queue
        struct job *job_executing = q->admitted_jobs[q->tail];
        q->admitted_jobs[q->tail] = NULL;
        q->num_admitted--;

        // wrap around if necessary
        q->tail = (q->tail + 1) % q->capacity;
        // Notify admission thread to admit as space free up
        pthread_cond_signal(&q->admission_cv);

        // release queue lock
        pthread_mutex_unlock(&q->lock);

        execute_single_job(job_executing);

        // Put job in processor record
        record(job_executing);
    }
    return NULL;
}
