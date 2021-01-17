#include "catch.hpp"
#include <cuda_complex.hpp>
#include <cmath>

using namespace complex;

TEST_CASE("complex_float is constructed from initializer list") {
    complex::complex_float z = {1.0f, 2.0f};
    REQUIRE(std::abs(z.x - 1.0f) < 1e-15f);
    REQUIRE(std::abs(z.y - 2.0f) < 1e-15f);    
}

TEST_CASE("complex_float is constructed from cuFloatComplex") {
    cuFloatComplex c = make_cuFloatComplex(2.0f, -5.0f);
    complex::complex_float z(c);
    REQUIRE(std::abs(z.x - c.x) < 1e-15f);
    REQUIRE(std::abs(z.y - c.y) < 1e-15f);    
}

TEST_CASE("complex_float is convertable to cuFloatComplex") {
    complex::complex_float z = {1.0f, 2.0f};
    cuFloatComplex c = z;
    REQUIRE(std::abs(z.x - c.x) < 1e-15f);
    REQUIRE(std::abs(z.y - c.y) < 1e-15f);    
}

TEST_CASE("operator+ adds complex_float variables") {
    complex::complex_float a = {1.0f, 2.0f};
    complex::complex_float b = {-1.0f, -2.0f};
    complex::complex_float c = a + b;
    REQUIRE(std::abs(c.x) < 1e-15f);
    REQUIRE(std::abs(c.y) < 1e-15f);        
}

TEST_CASE("operator+ works with cuFloatComplex variables") {
    cuFloatComplex a = make_cuFloatComplex(2.0f, -5.0f);
    cuFloatComplex b = make_cuFloatComplex(-2.0f, 5.0f);
    complex::complex_float c;
    c = a + b;
    REQUIRE(std::abs(c.x) < 1e-15f);
    REQUIRE(std::abs(c.y) < 1e-15f);        
}

TEST_CASE("operator+ works with mixed variables") {
    cuFloatComplex a = make_cuFloatComplex(2.0f, -5.0f);
    complex::complex_float b(-2.0f, 5.0f);    
    complex::complex_float c;
    c = a + b;
    REQUIRE(std::abs(c.x) < 1e-15f);
    REQUIRE(std::abs(c.y) < 1e-15f);        
}


TEST_CASE("assignment works with cuFloatComplex variables") {
    cuFloatComplex a = make_cuFloatComplex(2.0f, -5.0f);
    complex::complex_float c;
    c = a;
    REQUIRE(std::abs(c.x - a.x) < 1e-15f);
    REQUIRE(std::abs(c.y - a.y) < 1e-15f);        
}

TEST_CASE("operator+ can be chained") {
    complex::complex_float a = {1.0f, 2.0f};
    complex::complex_float b = {-1.0f, -2.0f};
    complex::complex_float c = a + b + a + b;
    REQUIRE(std::abs(c.x) < 1e-15f);
    REQUIRE(std::abs(c.y) < 1e-15f);        
}

TEST_CASE("real values can be added to complex_float") {
    complex::complex_float a = {1.0f, 2.0f};
    const float b = -1.0f;
    complex::complex_float c = a + b;
    REQUIRE(std::abs(c.x) < 1e-15f);
    complex::complex_float d = b + a;
    REQUIRE(std::abs(d.x) < 1e-15f);

}