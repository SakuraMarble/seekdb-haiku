/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include "gpu_worker.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  cuvsCagraSearchParams_t search_params;
  pthread_mutex_t search_mutex;
  void *query_buffer, *neighbor_buffer, *distance_buffer;
  size_t query_capacity, neighbor_capacity, distance_capacity;
  void *host_buffer;
  size_t host_capacity;
  long n, dim;
} sk_cuvs_index;

/* Grow scratch space on demand. Index-owned buffers avoid three device
 * allocations and frees for every single-row SQL search. */
static int reserve_buffer(sk_cuvs_index *h, void **buffer, size_t *capacity, size_t bytes)
{
  if (bytes <= *capacity) return 0;
  void *replacement = NULL;
  if (cuvsRMMAlloc(h->res, &replacement, bytes) != CUVS_SUCCESS) return 1;
  if (*buffer) cuvsRMMFree(h->res, *buffer, *capacity);
  *buffer = replacement;
  *capacity = bytes;
  return 0;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (pthread_mutex_init(&h->search_mutex, NULL) != 0) { free(h); return NULL; }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) {
    pthread_mutex_destroy(&h->search_mutex); free(h); return NULL;
  }

  /* IVF-PQ's internal low-precision arithmetic loses neighbours on large
   * descriptor values. A power-of-two scale preserves L2 neighbour order
   * without introducing rounding in normal FP32 values. Only graph training
   * uses the scaled copy; search and snapshots retain the original data. */
  float *scaled = NULL;
  float maximum = 1.0f;
  const size_t elements = (size_t)n * dim;
  for (size_t i = 0; i < elements; ++i) {
    const float value = fabsf(base[i]);
    if (value > maximum) maximum = value;
  }
  if (maximum > 1.0f) {
    int exponent = 0;
    frexpf(maximum, &exponent);
    const float scale = ldexpf(1.0f, -exponent);
    scaled = (float*)malloc(elements * sizeof(float));
    if (!scaled) goto fail_res;
    for (size_t i = 0; i < elements; ++i) scaled[i] = base[i] * scale;
  }

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)(scaled ? scaled : base);
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  h->ip->build_algo = IVF_PQ;
  h->ip->graph_degree = 64;
  h->ip->intermediate_graph_degree = 128;
  h->ip->nn_descent_niter = 100;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  if (scaled) {
    DLManagedTensor dataset = {0};
    if (cuvsCagraIndexGetDataset(h->idx, &dataset) != CUVS_SUCCESS) goto fail_idx;
    cudaStream_t stream = 0;
    int failed = cuvsStreamGet(h->res, &stream) != CUVS_SUCCESS;
    const size_t row_bytes = (size_t)dim * sizeof(float);
    const int64_t row_stride = dataset.dl_tensor.strides ? dataset.dl_tensor.strides[0] : dim;
    char *device = (char*)dataset.dl_tensor.data + dataset.dl_tensor.byte_offset;
    if (!failed && row_stride == dim) {
      failed = cudaMemcpyAsync(device, base, elements * sizeof(float),
                               cudaMemcpyHostToDevice, stream) != cudaSuccess;
    } else if (!failed) {
      for (long row = 0; row < n && !failed; ++row) {
        failed = cudaMemcpyAsync(device + row * row_stride * sizeof(float),
                                 base + row * dim, row_bytes,
                                 cudaMemcpyHostToDevice, stream) != cudaSuccess;
      }
    }
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS) failed = 1;
    if (dataset.deleter) dataset.deleter(&dataset);
    if (failed) goto fail_idx;
  }
  free(scaled); scaled = NULL;
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsStreamSync(h->res); cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsStreamSync(h->res); free(scaled); cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->search_mutex); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

static int search_on_worker(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !query || !out_ids || nq <= 0 || topk <= 0 || topk > h->n) return 1;
  int rc = 0;
  float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  const size_t query_bytes = sizeof(float)*nq*h->dim;
  const size_t result_bytes = sizeof(unsigned)*nq*topk;
  const size_t host_bytes = query_bytes + result_bytes + sizeof(float)*nq*topk;
  void *replacement = NULL;
  char *host_query = NULL, *host_ids = NULL, *host_distances = NULL;
  cudaStream_t stream = 0;
  int stream_ready = 0;
  pthread_mutex_lock(&h->search_mutex);
  if (reserve_buffer(h, &h->query_buffer, &h->query_capacity, sizeof(float)*nq*h->dim)) { rc=10; goto done; }
  if (reserve_buffer(h, &h->neighbor_buffer, &h->neighbor_capacity, sizeof(unsigned)*nq*topk)) { rc=11; goto done; }
  if (reserve_buffer(h, &h->distance_buffer, &h->distance_capacity, sizeof(float)*nq*topk)) { rc=12; goto done; }
  qd = (float*)h->query_buffer; nd = (unsigned*)h->neighbor_buffer; dd = (float*)h->distance_buffer;
  if (host_bytes > h->host_capacity) {
#ifdef __HIP_PLATFORM_AMD__
    if (cudaMallocHost(&replacement, host_bytes, 0) != cudaSuccess) { rc=17; goto done; }
#else
    if (cudaMallocHost(&replacement, host_bytes) != cudaSuccess) { rc=17; goto done; }
#endif
    if (h->host_buffer) cudaFreeHost(h->host_buffer);
    h->host_buffer = replacement; h->host_capacity = host_bytes;
  }
  host_query = (char*)h->host_buffer;
  host_ids = host_query + query_bytes;
  host_distances = host_ids + result_bytes;
  memcpy(host_query, query, query_bytes);
  if (cuvsStreamGet(h->res, &stream) != CUVS_SUCCESS) { rc=18; goto done; }
  stream_ready = 1;
  if (cudaMemcpyAsync(qd, host_query, query_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) { rc=13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    if (!h->search_params && cuvsCagraSearchParamsCreate(&h->search_params) != CUVS_SUCCESS) { rc=14; goto done; }
    sp = h->search_params;
    {
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 4096) required_itopk = 4096;
      sp->itopk_size = required_itopk;
      sp->search_width = 4;
      sp->algo = AUTO;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpyAsync(host_ids, nd, result_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) { rc=16; goto done; }
    if (out_dist && cudaMemcpyAsync(host_distances, dd, sizeof(float)*nq*topk, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      rc=16; goto done;
    }
  }
done:
  /* One stream barrier covers H2D, search, and D2H. It also makes scratch reuse
   * safe after an intermediate failure; all caller outputs remain synchronous. */
  if (stream_ready && cuvsStreamSync(h->res) != CUVS_SUCCESS && rc == 0) rc=19;
  if (rc == 0) {
    memcpy(out_ids, host_ids, result_bytes);
    if (out_dist) memcpy(out_dist, host_distances, sizeof(float)*nq*topk);
  }
  pthread_mutex_unlock(&h->search_mutex);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

typedef struct {
  void *handle;
  const float *query;
  long nq, topk;
  unsigned int *ids;
  float *distances;
  int result;
} search_work;

static void run_search(void *argument)
{
  search_work *work = (search_work*)argument;
  work->result = search_on_worker(work->handle, work->query, work->nq,
                                 work->topk, work->ids, work->distances);
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  search_work work = {handle, query, nq, topk, out_ids, out_dist, -1};
  if (gpu_worker_run(run_search, &work) != 0) return -1;
  return work.result;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->query_buffer) cuvsRMMFree(h->res, h->query_buffer, h->query_capacity);
  if (h->neighbor_buffer) cuvsRMMFree(h->res, h->neighbor_buffer, h->neighbor_capacity);
  if (h->distance_buffer) cuvsRMMFree(h->res, h->distance_buffer, h->distance_capacity);
  if (h->host_buffer) cudaFreeHost(h->host_buffer);
  if (h->search_params) cuvsCagraSearchParamsDestroy(h->search_params);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->search_mutex);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (pthread_mutex_init(&index->search_mutex, NULL) != 0) { free(index); return NULL; }
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  pthread_mutex_destroy(&index->search_mutex); free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
