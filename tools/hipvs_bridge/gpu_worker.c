/* Copyright (c) 2026 OceanBase.
 * Licensed under the Apache License, Version 2.0. */
#include "gpu_worker.h"
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct work_item {
  void (*function)(void *);
  void *argument;
  struct work_item *next;
  pthread_cond_t completed;
  int done;
} work_item;

static pthread_once_t initialized = PTHREAD_ONCE_INIT;
static pthread_once_t exit_registered = PTHREAD_ONCE_INIT;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t available = PTHREAD_COND_INITIALIZER;
static pthread_t worker;
static work_item *head = NULL, *tail = NULL;
static int startup_error = EAGAIN;
static int stopping = 0;

static void stop_worker(void);

static void register_exit(void)
{
  /* Register after the first callback: GPU libraries initialize lazy globals
   * there. Join this thread before their process-exit destructors release them. */
  atexit(stop_worker);
}

static void *worker_loop(void *unused)
{
  (void)unused;
  pthread_mutex_lock(&mutex);
  for (;;) {
    while (head == NULL && !stopping) pthread_cond_wait(&available, &mutex);
    if (head == NULL && stopping) break;
    work_item *item = head;
    head = item->next;
    if (head == NULL) tail = NULL;
    pthread_mutex_unlock(&mutex);
    item->function(item->argument);
    pthread_once(&exit_registered, register_exit);
    pthread_mutex_lock(&mutex);
    item->done = 1;
    pthread_cond_signal(&item->completed);
  }
  pthread_mutex_unlock(&mutex);
  return NULL;
}

static void start_worker(void)
{
  pthread_attr_t attributes;
  startup_error = pthread_attr_init(&attributes);
  if (startup_error != 0) return;
  startup_error = pthread_attr_setstacksize(&attributes, 32UL * 1024 * 1024);
  if (startup_error == 0) startup_error = pthread_create(&worker, &attributes, worker_loop, NULL);
  pthread_attr_destroy(&attributes);
}

int gpu_worker_run(void (*function)(void *), void *argument)
{
  pthread_once(&initialized, start_worker);
  if (startup_error != 0) return startup_error;
  if (pthread_equal(pthread_self(), worker)) {
    function(argument);
    return 0;
  }
  work_item item = {function, argument, NULL, PTHREAD_COND_INITIALIZER, 0};
  pthread_mutex_lock(&mutex);
  if (stopping) {
    pthread_mutex_unlock(&mutex);
    pthread_cond_destroy(&item.completed);
    return ESHUTDOWN;
  }
  if (tail == NULL) head = &item;
  else tail->next = &item;
  tail = &item;
  pthread_cond_signal(&available);
  while (!item.done) pthread_cond_wait(&item.completed, &mutex);
  pthread_mutex_unlock(&mutex);
  pthread_cond_destroy(&item.completed);
  return 0;
}

/* dlclose must not leave a thread executing instructions from this library. */
__attribute__((destructor)) static void stop_worker(void)
{
  if (startup_error != 0) return;
  pthread_mutex_lock(&mutex);
  if (stopping) {
    pthread_mutex_unlock(&mutex);
    return;
  }
  stopping = 1;
  pthread_cond_signal(&available);
  pthread_mutex_unlock(&mutex);
  pthread_join(worker, NULL);
}
