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

CC := gcc
null :=
space := $(null) $(null)
WEBOTS_HOME_PATH?=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))

INCLUDE_PATHS += -I/home/chryssida/darknet/include

LIBRARY_PATHS += -L/home/chryssida/darknet

LIBRARIES += -ldarknet -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio 

include $(WEBOTS_HOME_PATH)/resources/Makefile.os.include

TARGETS = autonomous_vehicle.Makefile \
	boomer.Makefile \
	radar_target_tracker.Makefile \
	racing_wheel.Makefile \
	VehicleDriver.Makefile

ifeq ($(OSTYPE),linux)
TARGETS += ros_automobile.Makefile
endif

.PHONY: release debug profile clean

release debug profile clean: $(TARGETS)

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

$(EXECUTABLE): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@


%.Makefile:
	+@echo "# make" $(MAKECMDGOALS) $(@:.Makefile=)
	+@echo "# CFLAGS" $(CFLAGS)
	+@make -s -C $(@:.Makefile=) $(MAKECMDGOALS)
	
clean:
	rm -f $(OBJS) $(EXECUTABLE)
