#include "catch.hpp"
#include "buffer.hpp" 
#include <cmath>

template <typename T>
void all_close(const T* x, const T* y, int size, T tol) {
    for (int k=0; k<size; ++k) {
        REQUIRE(std::abs(x[k] - y[k]) < tol);
    }
}


TEST_CASE("Buffer can be created when size is provided") {
    auto construct = [&](int s){complex::containers::buffer<float> b(s);};
    const int size = 12;
    REQUIRE_NOTHROW(construct(size));
}

TEST_CASE("Correct buffer size is returned") {
    const int size = 12;
    complex::containers::buffer<float> b(size);
    REQUIRE(b.size() == size);
}

TEST_CASE("Buffer can be created from data") {
    const int size = 4;
    const float data[size] = {1.0f, 2.0f, 3.0f, 4.0f};
    complex::containers::buffer<float> b(data, size);
    SECTION("and buffer size is correct") {
        REQUIRE(b.size() == size);
    }
    SECTION("and content is correct") {
        float stored[size];
        b.copy_to(stored, size);
        all_close(data, stored, size, 1e-15f);  
    }
}

TEST_CASE("Data can be copied to the buffer") {
    const int size = 3;
    const float data[size] = {1.0f, 2.0f, 3.0f};
    complex::containers::buffer<float> b(size);
    b.copy_from(data, size);
    float stored[size];
    b.copy_to(stored, size);
    all_close(data, stored, size, 1e-15f);
}

TEST_CASE("Copy operator works as expected") {
    const int size = 3;
    const int data[size] = {1, 2, -3};
    complex::containers::buffer<int> b0(data, size);
    complex::containers::buffer<int> b1(b0);
    int stored[size];
    b1.copy_to(stored, size);
    all_close(data, stored, size, 1);
}

TEST_CASE("Move constructor assigns pointer to source memory buffer") {
    const int size = 2;
    const int data[size] = {101, 23};
    complex::containers::buffer<int> a(data, size);
    const int* a_ptr = a;
    complex::containers::buffer<int> b(std::move(a));
    const int* b_ptr = b;
    REQUIRE(a_ptr == b_ptr);
    REQUIRE(a.size() == 0);
    
    const int* a_after = a;
    REQUIRE(a_after == nullptr);
}