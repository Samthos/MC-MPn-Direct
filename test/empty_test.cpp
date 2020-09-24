#include "gtest/gtest.h"


// Class is used to name fixture
// class is used to name instance of fixture
// CLASS is used to name testing::Test class


namespace {
  template <template <typename, typename> typename Container, template <typename> typename Allocator>
  class Class_Fixture {
    public:
      Class_Fixture() {}
  };


  template <typename T>
  class CLASSTest : public testing::Test {
   public:
     T class_fixture;
  };

  using Implementations = testing::Types<
    Class_Fixture<std::vector, std::allocator>
    >;
  TYPED_TEST_SUITE(CLASSTest, Implementations);

  TYPED_TEST(CLASSTest, EmptyTest) { }
}
