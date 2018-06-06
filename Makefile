# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/sina/github/lda_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/sina/github/lda_demo

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/Applications/CLion.app/Contents/bin/cmake/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/Applications/CLion.app/Contents/bin/cmake/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/sina/github/lda_demo/CMakeFiles /Users/sina/github/lda_demo/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/sina/github/lda_demo/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named lda_demo

# Build rule for target.
lda_demo: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 lda_demo
.PHONY : lda_demo

# fast build rule for target.
lda_demo/fast:
	$(MAKE) -f CMakeFiles/lda_demo.dir/build.make CMakeFiles/lda_demo.dir/build
.PHONY : lda_demo/fast

#=============================================================================
# Target rules for targets named gtest

# Build rule for target.
gtest: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gtest
.PHONY : gtest

# fast build rule for target.
gtest/fast:
	$(MAKE) -f thirdparty/cppjieba/deps/gtest/CMakeFiles/gtest.dir/build.make thirdparty/cppjieba/deps/gtest/CMakeFiles/gtest.dir/build
.PHONY : gtest/fast

#=============================================================================
# Target rules for targets named load_test

# Build rule for target.
load_test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 load_test
.PHONY : load_test

# fast build rule for target.
load_test/fast:
	$(MAKE) -f thirdparty/cppjieba/test/CMakeFiles/load_test.dir/build.make thirdparty/cppjieba/test/CMakeFiles/load_test.dir/build
.PHONY : load_test/fast

#=============================================================================
# Target rules for targets named demo

# Build rule for target.
demo: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 demo
.PHONY : demo

# fast build rule for target.
demo/fast:
	$(MAKE) -f thirdparty/cppjieba/test/CMakeFiles/demo.dir/build.make thirdparty/cppjieba/test/CMakeFiles/demo.dir/build
.PHONY : demo/fast

#=============================================================================
# Target rules for targets named test.run

# Build rule for target.
test.run: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test.run
.PHONY : test.run

# fast build rule for target.
test.run/fast:
	$(MAKE) -f thirdparty/cppjieba/test/unittest/CMakeFiles/test.run.dir/build.make thirdparty/cppjieba/test/unittest/CMakeFiles/test.run.dir/build
.PHONY : test.run/fast

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/lda_demo.dir/build.make CMakeFiles/lda_demo.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/lda_demo.dir/build.make CMakeFiles/lda_demo.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/lda_demo.dir/build.make CMakeFiles/lda_demo.dir/main.cpp.s
.PHONY : main.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... lda_demo"
	@echo "... gtest"
	@echo "... load_test"
	@echo "... demo"
	@echo "... test.run"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

