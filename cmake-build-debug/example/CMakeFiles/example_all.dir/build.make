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
include example/CMakeFiles/example_all.dir/depend.make

# Include the progress variables for this target.
include example/CMakeFiles/example_all.dir/progress.make

# Include the compile flags for this target's objects.
include example/CMakeFiles/example_all.dir/flags.make

example/CMakeFiles/example_all.dir/dummy.example.cc.o: example/CMakeFiles/example_all.dir/flags.make
example/CMakeFiles/example_all.dir/dummy.example.cc.o: ../example/dummy.example.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object example/CMakeFiles/example_all.dir/dummy.example.cc.o"
	cd /home/karl/Documents/tensors/cmake-build-debug/example && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_all.dir/dummy.example.cc.o -c /home/karl/Documents/tensors/example/dummy.example.cc

example/CMakeFiles/example_all.dir/dummy.example.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_all.dir/dummy.example.cc.i"
	cd /home/karl/Documents/tensors/cmake-build-debug/example && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karl/Documents/tensors/example/dummy.example.cc > CMakeFiles/example_all.dir/dummy.example.cc.i

example/CMakeFiles/example_all.dir/dummy.example.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_all.dir/dummy.example.cc.s"
	cd /home/karl/Documents/tensors/cmake-build-debug/example && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karl/Documents/tensors/example/dummy.example.cc -o CMakeFiles/example_all.dir/dummy.example.cc.s

# Object files for target example_all
example_all_OBJECTS = \
"CMakeFiles/example_all.dir/dummy.example.cc.o"

# External object files for target example_all
example_all_EXTERNAL_OBJECTS =

example/example_all: example/CMakeFiles/example_all.dir/dummy.example.cc.o
example/example_all: example/CMakeFiles/example_all.dir/build.make
example/example_all: /usr/lib/x86_64-linux-gnu/libSM.so
example/example_all: /usr/lib/x86_64-linux-gnu/libICE.so
example/example_all: /usr/lib/x86_64-linux-gnu/libX11.so
example/example_all: /usr/lib/x86_64-linux-gnu/libXext.so
example/example_all: /usr/lib/x86_64-linux-gnu/libOpenCL.so
example/example_all: example/CMakeFiles/example_all.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/karl/Documents/tensors/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_all"
	cd /home/karl/Documents/tensors/cmake-build-debug/example && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_all.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example/CMakeFiles/example_all.dir/build: example/example_all

.PHONY : example/CMakeFiles/example_all.dir/build

example/CMakeFiles/example_all.dir/clean:
	cd /home/karl/Documents/tensors/cmake-build-debug/example && $(CMAKE_COMMAND) -P CMakeFiles/example_all.dir/cmake_clean.cmake
.PHONY : example/CMakeFiles/example_all.dir/clean

example/CMakeFiles/example_all.dir/depend:
	cd /home/karl/Documents/tensors/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/karl/Documents/tensors /home/karl/Documents/tensors/example /home/karl/Documents/tensors/cmake-build-debug /home/karl/Documents/tensors/cmake-build-debug/example /home/karl/Documents/tensors/cmake-build-debug/example/CMakeFiles/example_all.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : example/CMakeFiles/example_all.dir/depend

