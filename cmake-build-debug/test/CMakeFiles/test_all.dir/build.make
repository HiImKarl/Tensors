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
include test/CMakeFiles/test_all.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_all.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_all.dir/flags.make

test/CMakeFiles/test_all.dir/access.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/access.test.cc.o: ../test/access.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_all.dir/access.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/access.test.cc.o -c /home/karl/Documents/tensors/test/access.test.cc

test/CMakeFiles/test_all.dir/access.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/access.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/access.test.cc > CMakeFiles/test_all.dir/access.test.cc.i

test/CMakeFiles/test_all.dir/access.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/access.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/access.test.cc -o CMakeFiles/test_all.dir/access.test.cc.s

test/CMakeFiles/test_all.dir/arithmetic.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/arithmetic.test.cc.o: ../test/arithmetic.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/test_all.dir/arithmetic.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/arithmetic.test.cc.o -c /home/karl/Documents/tensors/test/arithmetic.test.cc

test/CMakeFiles/test_all.dir/arithmetic.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/arithmetic.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/arithmetic.test.cc > CMakeFiles/test_all.dir/arithmetic.test.cc.i

test/CMakeFiles/test_all.dir/arithmetic.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/arithmetic.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/arithmetic.test.cc -o CMakeFiles/test_all.dir/arithmetic.test.cc.s

test/CMakeFiles/test_all.dir/assign.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/assign.test.cc.o: ../test/assign.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object test/CMakeFiles/test_all.dir/assign.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/assign.test.cc.o -c /home/karl/Documents/tensors/test/assign.test.cc

test/CMakeFiles/test_all.dir/assign.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/assign.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/assign.test.cc > CMakeFiles/test_all.dir/assign.test.cc.i

test/CMakeFiles/test_all.dir/assign.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/assign.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/assign.test.cc -o CMakeFiles/test_all.dir/assign.test.cc.s

test/CMakeFiles/test_all.dir/containers.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/containers.test.cc.o: ../test/containers.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object test/CMakeFiles/test_all.dir/containers.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/containers.test.cc.o -c /home/karl/Documents/tensors/test/containers.test.cc

test/CMakeFiles/test_all.dir/containers.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/containers.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/containers.test.cc > CMakeFiles/test_all.dir/containers.test.cc.i

test/CMakeFiles/test_all.dir/containers.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/containers.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/containers.test.cc -o CMakeFiles/test_all.dir/containers.test.cc.s

test/CMakeFiles/test_all.dir/expressions.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/expressions.test.cc.o: ../test/expressions.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object test/CMakeFiles/test_all.dir/expressions.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/expressions.test.cc.o -c /home/karl/Documents/tensors/test/expressions.test.cc

test/CMakeFiles/test_all.dir/expressions.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/expressions.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/expressions.test.cc > CMakeFiles/test_all.dir/expressions.test.cc.i

test/CMakeFiles/test_all.dir/expressions.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/expressions.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/expressions.test.cc -o CMakeFiles/test_all.dir/expressions.test.cc.s

test/CMakeFiles/test_all.dir/intialize.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/intialize.test.cc.o: ../test/intialize.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object test/CMakeFiles/test_all.dir/intialize.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/intialize.test.cc.o -c /home/karl/Documents/tensors/test/intialize.test.cc

test/CMakeFiles/test_all.dir/intialize.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/intialize.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/intialize.test.cc > CMakeFiles/test_all.dir/intialize.test.cc.i

test/CMakeFiles/test_all.dir/intialize.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/intialize.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/intialize.test.cc -o CMakeFiles/test_all.dir/intialize.test.cc.s

test/CMakeFiles/test_all.dir/iterator.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/iterator.test.cc.o: ../test/iterator.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object test/CMakeFiles/test_all.dir/iterator.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/iterator.test.cc.o -c /home/karl/Documents/tensors/test/iterator.test.cc

test/CMakeFiles/test_all.dir/iterator.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/iterator.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/iterator.test.cc > CMakeFiles/test_all.dir/iterator.test.cc.i

test/CMakeFiles/test_all.dir/iterator.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/iterator.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/iterator.test.cc -o CMakeFiles/test_all.dir/iterator.test.cc.s

test/CMakeFiles/test_all.dir/manip.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/manip.test.cc.o: ../test/manip.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object test/CMakeFiles/test_all.dir/manip.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/manip.test.cc.o -c /home/karl/Documents/tensors/test/manip.test.cc

test/CMakeFiles/test_all.dir/manip.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/manip.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/manip.test.cc > CMakeFiles/test_all.dir/manip.test.cc.i

test/CMakeFiles/test_all.dir/manip.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/manip.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/manip.test.cc -o CMakeFiles/test_all.dir/manip.test.cc.s

test/CMakeFiles/test_all.dir/object.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/object.test.cc.o: ../test/object.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object test/CMakeFiles/test_all.dir/object.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/object.test.cc.o -c /home/karl/Documents/tensors/test/object.test.cc

test/CMakeFiles/test_all.dir/object.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/object.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/object.test.cc > CMakeFiles/test_all.dir/object.test.cc.i

test/CMakeFiles/test_all.dir/object.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/object.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/object.test.cc -o CMakeFiles/test_all.dir/object.test.cc.s

test/CMakeFiles/test_all.dir/opencl.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/opencl.test.cc.o: ../test/opencl.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object test/CMakeFiles/test_all.dir/opencl.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/opencl.test.cc.o -c /home/karl/Documents/tensors/test/opencl.test.cc

test/CMakeFiles/test_all.dir/opencl.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/opencl.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/opencl.test.cc > CMakeFiles/test_all.dir/opencl.test.cc.i

test/CMakeFiles/test_all.dir/opencl.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/opencl.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/opencl.test.cc -o CMakeFiles/test_all.dir/opencl.test.cc.s

test/CMakeFiles/test_all.dir/shape.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/shape.test.cc.o: ../test/shape.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object test/CMakeFiles/test_all.dir/shape.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/shape.test.cc.o -c /home/karl/Documents/tensors/test/shape.test.cc

test/CMakeFiles/test_all.dir/shape.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/shape.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/shape.test.cc > CMakeFiles/test_all.dir/shape.test.cc.i

test/CMakeFiles/test_all.dir/shape.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/shape.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/shape.test.cc -o CMakeFiles/test_all.dir/shape.test.cc.s

test/CMakeFiles/test_all.dir/slice.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/slice.test.cc.o: ../test/slice.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object test/CMakeFiles/test_all.dir/slice.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/slice.test.cc.o -c /home/karl/Documents/tensors/test/slice.test.cc

test/CMakeFiles/test_all.dir/slice.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/slice.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/slice.test.cc > CMakeFiles/test_all.dir/slice.test.cc.i

test/CMakeFiles/test_all.dir/slice.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/slice.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/slice.test.cc -o CMakeFiles/test_all.dir/slice.test.cc.s

test/CMakeFiles/test_all.dir/string.test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/string.test.cc.o: ../test/string.test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object test/CMakeFiles/test_all.dir/string.test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/string.test.cc.o -c /home/karl/Documents/tensors/test/string.test.cc

test/CMakeFiles/test_all.dir/string.test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/string.test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/string.test.cc > CMakeFiles/test_all.dir/string.test.cc.i

test/CMakeFiles/test_all.dir/string.test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/string.test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/string.test.cc -o CMakeFiles/test_all.dir/string.test.cc.s

test/CMakeFiles/test_all.dir/test.cc.o: test/CMakeFiles/test_all.dir/flags.make
test/CMakeFiles/test_all.dir/test.cc.o: ../test/test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object test/CMakeFiles/test_all.dir/test.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_all.dir/test.cc.o -c /home/karl/Documents/tensors/test/test.cc

test/CMakeFiles/test_all.dir/test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_all.dir/test.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/test/test.cc > CMakeFiles/test_all.dir/test.cc.i

test/CMakeFiles/test_all.dir/test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_all.dir/test.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/test/test.cc -o CMakeFiles/test_all.dir/test.cc.s

# Object files for target test_all
test_all_OBJECTS = \
"CMakeFiles/test_all.dir/access.test.cc.o" \
"CMakeFiles/test_all.dir/arithmetic.test.cc.o" \
"CMakeFiles/test_all.dir/assign.test.cc.o" \
"CMakeFiles/test_all.dir/containers.test.cc.o" \
"CMakeFiles/test_all.dir/expressions.test.cc.o" \
"CMakeFiles/test_all.dir/intialize.test.cc.o" \
"CMakeFiles/test_all.dir/iterator.test.cc.o" \
"CMakeFiles/test_all.dir/manip.test.cc.o" \
"CMakeFiles/test_all.dir/object.test.cc.o" \
"CMakeFiles/test_all.dir/opencl.test.cc.o" \
"CMakeFiles/test_all.dir/shape.test.cc.o" \
"CMakeFiles/test_all.dir/slice.test.cc.o" \
"CMakeFiles/test_all.dir/string.test.cc.o" \
"CMakeFiles/test_all.dir/test.cc.o"

# External object files for target test_all
test_all_EXTERNAL_OBJECTS =

test/all: test/CMakeFiles/test_all.dir/access.test.cc.o
test/all: test/CMakeFiles/test_all.dir/arithmetic.test.cc.o
test/all: test/CMakeFiles/test_all.dir/assign.test.cc.o
test/all: test/CMakeFiles/test_all.dir/containers.test.cc.o
test/all: test/CMakeFiles/test_all.dir/expressions.test.cc.o
test/all: test/CMakeFiles/test_all.dir/intialize.test.cc.o
test/all: test/CMakeFiles/test_all.dir/iterator.test.cc.o
test/all: test/CMakeFiles/test_all.dir/manip.test.cc.o
test/all: test/CMakeFiles/test_all.dir/object.test.cc.o
test/all: test/CMakeFiles/test_all.dir/opencl.test.cc.o
test/all: test/CMakeFiles/test_all.dir/shape.test.cc.o
test/all: test/CMakeFiles/test_all.dir/slice.test.cc.o
test/all: test/CMakeFiles/test_all.dir/string.test.cc.o
test/all: test/CMakeFiles/test_all.dir/test.cc.o
test/all: test/CMakeFiles/test_all.dir/build.make
test/all: /usr/lib/x86_64-linux-gnu/libOpenCL.so
test/all: test/CMakeFiles/test_all.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Linking CXX executable all"
	cd /home/karl/Documents/tensors/cmake-build-debug/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_all.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_all.dir/build: test/all

.PHONY : test/CMakeFiles/test_all.dir/build

test/CMakeFiles/test_all.dir/clean:
	cd /home/karl/Documents/tensors/cmake-build-debug/test && $(CMAKE_COMMAND) -P CMakeFiles/test_all.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_all.dir/clean

test/CMakeFiles/test_all.dir/depend:
	cd /home/karl/Documents/tensors/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/karl/Documents/tensors /home/karl/Documents/tensors/test /home/karl/Documents/tensors/cmake-build-debug /home/karl/Documents/tensors/cmake-build-debug/test /home/karl/Documents/tensors/cmake-build-debug/test/CMakeFiles/test_all.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_all.dir/depend

