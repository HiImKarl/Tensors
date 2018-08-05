#include <iostream>
#include <string>

inline std::string BeginTest(std::string const &test, std::string const &container) 
{
  std::cout << "Running... | " + test + " | " + container << std::endl;
  return container + ": " + test;
}

inline std::string BeginTest(std::string const &test)
{
  std::cout << "Running... | " + test << std::endl;
  return test;
}
