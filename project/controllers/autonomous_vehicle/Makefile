# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Webots Makefile system
#
# You may add some variable definitions hereafter to customize the build process
# See documentation in $(WEBOTS_HOME_PATH)/resources/Makefile.include

CC := gcc
null :=
space := $(null) $(null)
WEBOTS_HOME_PATH?=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))

INCLUDE_PATHS += -I/home/chryssida/darknet/include
INCLUDE_PATHS += -I/usr/local/include/opencv4

LIBRARY_PATHS += -L/home/chryssida/darknet
LIBRARY_PATHS += -L/usr/local/lib

LIBRARIES = -ldriver -lcar -ldarknet -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio 

# Do not modify the following: this includes Webots global Makefile.include
include $(WEBOTS_HOME_PATH)/resources/Makefile.include

# Your Executable Name
EXECUTABLE = autonomous_vehicle

# C Compiler Flags
CFLAGS += $(INCLUDE_PATHS)

# Linker Flags
LDFLAGS += $(LIBRARY_PATHS) $(LIBRARIES)

# Your Source Files
SRCS = autonomous_vehicle.c

# Object Files
OBJS = $(SRCS:.c=.o)

# Make Rules
all: $(EXECUTABLE)

#$(EXECUTABLE): $(OBJS)
#	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@


	

