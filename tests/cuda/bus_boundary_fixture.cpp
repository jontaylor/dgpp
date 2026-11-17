// Compile the serving adapter with the host compiler: tp_bus.hpp also
// includes sampling/parser code that is not compiled by nvcc.
#include "engine/tp_bus.hpp"
#include <atomic>
#include <chrono>
#include <thread>

void test_launch_bulk_consumer(const uint16_t* data, cudaStream_t stream);

void test_stream_boundary_reduce(dgpp::net::CollectiveBus& bus, uint16_t* data, size_t elems, cudaStream_t stream) {
  dgpp::BusStreamReducer reducer(bus);
  reducer.bind_stream(stream);
  const char* enabled = std::getenv("DGPP_BULK_STREAM_ORDERED");
  if (enabled && std::atoi(enabled) && elems > 2 * dgpp::bus_latency_slot_elems(bus)) {
    uint16_t* mirror = nullptr;
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&mirror), elems * 2, cudaHostAllocDefault));
    struct Gate { std::atomic<bool> release{false}, timed_out{false}; } gate;
    DGPP_CUDA_OK(cudaLaunchHostFunc(stream, [](void* ptr) {
      auto& g = *static_cast<Gate*>(ptr);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!g.release.load()) {
        if (std::chrono::steady_clock::now() > deadline) { g.timed_out = true; break; }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }, &gate));
    // Submission must return while the producer is deliberately held back.
    try {
      reducer.reduce(data, static_cast<int>(elems / 2), 2);
      // Queue real consumer work before releasing the producer: a raw memory
      // wait alone does not describe this dependency to the CUDA scheduler.
      test_launch_bulk_consumer(data, stream);
      DGPP_CUDA_OK(cudaMemcpyAsync(mirror, data, elems * 2, cudaMemcpyDeviceToHost, stream));
    }
    catch (...) { gate.release = true; cudaStreamSynchronize(stream); throw; }
    gate.release = true;
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    reducer.settle();
    DGPP_CUDA_OK(cudaFreeHost(mirror));
    if (gate.timed_out) throw std::runtime_error("bulk submission blocked on producer readiness");
    std::vector<uint16_t> saved(elems), zeros(elems);
    DGPP_CUDA_OK(cudaMemcpyAsync(saved.data(), data, elems * 2, cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    // The same mapped word is reused without reset, with latency work between
    // bulk generations. Dependent D2H is queued before the host checks verdicts.
    DGPP_CUDA_OK(cudaMemsetAsync(data, 0, elems * 2, stream));
    reducer.reduce(data, 1, 2);
    reducer.reduce(data, static_cast<int>(elems / 2), 2);
    DGPP_CUDA_OK(cudaMemcpyAsync(zeros.data(), data, elems * 2, cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    reducer.settle();
    if (std::any_of(zeros.begin(), zeros.end(), [](auto x) { return x != 0; }))
      throw std::runtime_error("bulk completion reuse exposed an unfinished result");
    DGPP_CUDA_OK(cudaMemcpyAsync(data, saved.data(), elems * 2, cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  } else {
    reducer.reduce(data, static_cast<int>(elems / 2), 2);
  }
}

// Keep the producer unready until the bus watchdog fires. Failure must release
// a queued GPU wait as well as the CPU waiter, without publishing success.
void test_deferred_bulk_timeout(dgpp::net::CollectiveBus& bus) {
  cudaStream_t stream = nullptr;
  cudaEvent_t ready = nullptr;
  uint32_t* completion = nullptr;
  uint16_t* data = nullptr;
  const size_t elems = 2 * dgpp::bus_latency_slot_elems(bus) + 2;
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&completion), sizeof(uint32_t), cudaHostAllocMapped));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&data), elems * 2));
  *completion = 0;
  void* device_completion = nullptr;
  DGPP_CUDA_OK(cudaHostGetDevicePointer(&device_completion, completion, 0));
  std::atomic<bool> release{false};
  DGPP_CUDA_OK(cudaLaunchHostFunc(stream, [](void* ptr) {
    auto& released = *static_cast<std::atomic<bool>*>(ptr);
    while (!released.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
  }, &release));
  DGPP_CUDA_OK(cudaEventRecord(ready, stream));
  std::string error;
  const auto id = bus.allreduce_bulk(data, data, elems, &error, ready, completion, 1);
  const auto wait_status = dgpp::net::launch_bus_completion_wait(
      static_cast<const uint32_t*>(device_completion), 1, stream);
  const auto result = bus.wait_allreduce(id, 10000);
  const bool signalled = __atomic_load_n(completion, __ATOMIC_ACQUIRE) == 1;
  // Release even a defective implementation so the failure is reported, not hung.
  __atomic_store_n(completion, 1u, __ATOMIC_RELEASE);
  release = true;
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  cudaEventDestroy(ready);
  cudaFreeHost(completion);
  cudaFree(data);
  cudaStreamDestroy(stream);
  if (!id || wait_status != cudaSuccess || result.ok || !signalled ||
      result.error.find("completion timeout") == std::string::npos)
    throw std::runtime_error("deferred bulk timeout did not fail and release both waiters: " + result.error);
}
