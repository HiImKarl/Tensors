# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/83/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/83/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/karl/Documents/tensors

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/karl/Documents/tensors/cmake-build-debug

# Include any dependencies generated for this target.
include test/CMakeFiles/test_slice.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_slice.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_slice.dir/flags.make

test/CMakeFiles/test_slice.dir/slice.test.cc.o: test/CMakeFiles/test_slice.dir/flags.make
test/CMakeFiles/test_slice.dir/slice.test.cc.o: ../test/slice.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_slice.dir/slice.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_slice.dir/slice.test.cc.o -c /home/karl/Documents/tensors/test/slice.test.cc

test/CMakeFiles/test_slice.dir/slice.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_slice.dir/slice.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/slice.test.cc > CMakeFiles/test_slice.dir/slice.test.cc.i

test/CMakeFiles/test_slice.dir/slice.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_slice.dir/slice.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/slice.test.cc -o CMakeFiles/test_slice.dir/slice.test.cc.s

test/CMakeFiles/test_slice.dir/test.cc.o: test/CMakeFiles/test_slice.dir/flags.make
test/CMakeFiles/test_slice.dir/test.cc.o: ../test/test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/test_slice.dir/test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_slice.dir/test.cc.o -c /home/karl/Documents/tensors/test/test.cc

test/CMakeFiles/test_slice.dir/test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_slice.dir/test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/test.cc > CMakeFiles/test_slice.dir/test.cc.i

test/CMakeFiles/test_slice.dir/test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_slice.dir/test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/test.cc -o CMakeFiles/test_slice.dir/test.cc.s

# Object files for target test_slice
test_slice_OBJECTS = \
"CMakeFiles/test_slice.dir/slice.test.cc.o" \
"CMakeFiles/test_slice.dir/test.cc.o"

# External object files for target test_slice
test_slice_EXTERNAL_OBJECTS =

test/slice: test/CMakeFiles/test_slice.dir/slice.test.cc.o
test/slice: test/CMakeFiles/test_slice.dir/test.cc.o
test/slice: test/CMakeFiles/test_slice.dir/build.make
test/slice: /usr/lib/x86_64-linux-gnu/libOpenCL.so
test/slice: test/CMakeFiles/test_slice.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable slice"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_slice.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_slice.dir/build: test/slice

.PHONY : test/CMakeFiles/test_slice.dir/build

test/CMakeFiles/test_slice.dir/clean:
	cd /home/karl/Documents/tensors/cmake-build-debug/test && $(CMAKE_COMMAND) -P CMakeFiles/test_slice.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_slice.dir/clean

test/CMakeFiles/test_slice.dir/depend:
	cd /home/karl/Documents/tensors/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/karl/Documents/tensors /home/karl/Documents/tensors/test /home/karl/Documents/tensors/cmake-build-debug /home/karl/Documents/tensors/cmake-build-debug/test /home/karl/Documents/tensors/cmake-build-debug/test/CMakeFiles/test_slice.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_slice.dir/depend

