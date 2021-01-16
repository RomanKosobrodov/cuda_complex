#include <cuda_runtime.h>
#include <exception>
#include <algorithm>   // std::min

namespace complex {
namespace containers {
    template <typename T>
    class buffer {
        T* ptr;
        int buffer_size;
    public:    
        buffer()=delete;
        
        buffer(int size) 
        : ptr(nullptr)
        , buffer_size(size) {
            cudaError_t status = cudaMalloc(&ptr, size * sizeof(T));
            if (status != cudaSuccess) {
                throw std::runtime_error("Unable to allocate memory on device");
            }
        }

        buffer(const T* data, int size) : buffer(size) {
            cudaError_t status = cudaMemcpy(ptr, data, size * sizeof(T), cudaMemcpyDefault);
            if (status != cudaSuccess) {
                throw std::runtime_error("Unable to initialise buffer memory with data");
            }            
        }

        buffer(const buffer& other) : buffer(other.ptr, other.size()) {}

        buffer(buffer&& other)
        : ptr(other.ptr)
        , buffer_size(other.buffer_size)
        {
            other.ptr = nullptr;
            other.buffer_size = 0;
        }

        ~buffer(){
            cudaFree(ptr);
        }

        operator T*() {return ptr;}
        operator T*() const {return ptr;} 

        int size() const { return buffer_size; }

        void copy_to(T* destination, int size) const {
            const int copyBytes = sizeof(T) * std::min(size, buffer_size);
            cudaError_t status = cudaMemcpy(destination, ptr, copyBytes, cudaMemcpyDefault);
            if (status != cudaSuccess) {
                throw std::runtime_error("Unable to copy buffer content to destination");
            }            
        }

        void copy_from(const T* source, int size) {
            const int copyBytes = sizeof(T) * std::min(size, buffer_size);
            cudaError_t status = cudaMemcpy(ptr, source, copyBytes, cudaMemcpyDefault);
            if (status != cudaSuccess) {
                throw std::runtime_error("Unable to copy source to buffer");
            }             
        }        
    };
}
}