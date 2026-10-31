/* Copyright (c) 2026 OceanBase.
 * Licensed under the Apache License, Version 2.0. */
#include "gpu_worker.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>

typedef struct {
  int input, output;
  pthread_t executed_by;
} result;

static void calculate(void *argument)
{
  result *value = (result*)argument;
  value->output = value->input * 7;
  value->executed_by = pthread_self();
}

static void nested(void *argument)
{
  assert(gpu_worker_run(calculate, argument) == 0);
}

static void *caller(void *argument)
{
  intptr_t number = (intptr_t)argument;
  pthread_t previous;
  for (int i = 0; i < 200; ++i) {
    result value = {(int)(number * 200 + i), -1, pthread_self()};
    assert(gpu_worker_run(i % 2 == 0 ? calculate : nested, &value) == 0);
    assert(value.output == value.input * 7);
    assert(!pthread_equal(value.executed_by, pthread_self()));
    if (i != 0) assert(pthread_equal(previous, value.executed_by));
    previous = value.executed_by;
  }
  return NULL;
}

int main(void)
{
  pthread_t callers[8];
  for (intptr_t i = 0; i < 8; ++i) assert(pthread_create(&callers[i], NULL, caller, (void*)i) == 0);
  for (int i = 0; i < 8; ++i) assert(pthread_join(callers[i], NULL) == 0);
  return 0;
}
