/* Copyright (c) 2026 OceanBase.
 * Licensed under the Apache License, Version 2.0. */
#pragma once

/* Execute synchronously on one persistent 32 MiB-stack worker. The callback
 * finishes before return, so callers may pass stack-owned inputs and outputs. */
int gpu_worker_run(void (*function)(void *), void *argument);
