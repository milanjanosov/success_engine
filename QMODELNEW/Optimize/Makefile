
# For allowing png outputs during visualization, type this:
# make name_of_algo pngout=true
#
PNG_OUT := $(strip $(pngout))
DEBUG_MODE := $(strip $(debug))

DEFAULT_FLAGS := -lm
VIZUALIZER_FLAGS := -lGL -lGLU -lglut
PNGOUTPUT_FLAGS := -lIL -lILU -lILUT
ERROR_FLAGS := -Wall -Wextra
CANCEL_FLAGS := -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-switch -Wno-comment -Wno-unused-but-set-parameter -Wunused-function

##############

GCC := gcc src/robotsim_main.c src/utilities/datastructs.c src/utilities/dynamics_utils.c src/utilities/math_utils.c src/utilities/file_utils.c src/utilities/param_utils.c src/sensors.c src/robotmodel.c src/colors.c src/vizualizer/objects_2d.c src/objects_menu.c src/vizualizer/objects_3d.c src/stat.c src/dynspecviz.c src/utilities/output_utils.c src/utilities/debug_utils.c src/utilities/waypoints.c

# Setting up png output mode
ifeq ($(PNG_OUT), true)
 GCC += src/utilities/pngout_utils.c $(PNGOUTPUT_FLAGS) -DPNG_OUT
endif

# Debug mode for segfault detection
ifeq ($(DEBUG_MODE), true)
 GCC += -DDEBUG -rdynamic
endif

#GCC += $(DEFAULT_FLAGS) $(VIZUALIZER_FLAGS) $(ERROR_FLAGS) $(CANCEL_FLAGS) -o robotflocksim_main
GCC += $(DEFAULT_FLAGS) $(VIZUALIZER_FLAGS) $(ERROR_FLAGS) $(CANCEL_FLAGS) -o robotflocksim_main -pg

# Basic test model: agents connected with spring-like pair potentials
spring:
	$(GCC) src/algo_spring.c src/algo_spring_gui.c src/algo_spring_stat.c

# Chasing model
chasing:
	$(GCC) src/algo_chasing.c src/algo_chasing_gui.c src/algo_chasing_stat.c src/utilities/arenas.c

# Original SPP model with obstacles and shill agents
spp:
	$(GCC) src/algo_spp.c src/algo_spp_gui.c src/algo_spp_stat.c src/utilities/arenas.c src/utilities/obstacles.c

# 3D SPP model
spp_3D:
	$(GCC) src/algo_spp_3d.c src/algo_spp_3d_gui.c src/algo_spp_3d_stat.c

# SPP model for testing evolution algorithms
spp_evol:
	$(GCC) src/algo_spp_evol.c src/algo_spp_evol_gui.c src/algo_spp_evol_stat.c src/utilities/interactions.c src/utilities/arenas.c src/utilities/obstacles.c

# Hierarchical SPP model
spp_hier:
	$(GCC) src/algo_spp_hier.c src/algo_spp_hier_gui.c src/algo_spp_hier_stat.c src/utilities/interactions.c src/utilities/arenas.c src/utilities/obstacles.c

spp_int_hier_control:
	$(GCC) src/algo_spp_int_hier_control.c src/algo_spp_int_hier_control_gui.c src/algo_spp_int_hier_control_stat.c src/utilities/interactions.c src/utilities/arenas.c src/utilities/obstacles.c

# Steering algorithm based on Szabo et al
steering:
	$(GCC) src/algo_steering.c src/algo_steering_gui.c src/algo_spp_stat.c

# Target tracking algorithm
target_tracking:
	$(GCC) src/algo_target_tracking.c src/algo_target_tracking_gui.c src/algo_target_tracking_stat.c src/utilities/arenas.c

# Ring algorithm
ring:
	$(GCC) src/algo_ring.c src/algo_ring_gui.c src/algo_ring_stat.c src/utilities/interactions.c src/utilities/arenas.c

# Line algorithm
line:
	$(GCC) src/algo_line.c src/algo_line_gui.c src/algo_line_stat.c src/utilities/arenas.c

# Random target spreading model
random_targets:
	$(GCC) src/algo_random_targets.c src/algo_random_targets_gui.c src/algo_random_targets_stat.c src/utilities/arenas.c src/utilities/interactions.c

# "slowing down" individual-flocking model for dense UAV traffic simulation
traffic_slowdown:
	$(GCC) src/algo_traffic_slowdown.c src/algo_traffic_slowdown_gui.c src/algo_traffic_slowdown_stat.c src/traffic_targets.c src/utilities/interactions.c src/utilities/obstacles.c

# "roundabout" individual-flocking model for dense UAV traffic simulation
traffic_circle:
	$(GCC) src/algo_traffic_circle_stat.c src/algo_traffic_circle_gui.c src/algo_traffic_circle.c src/traffic_targets.c src/utilities/interactions.c

# collective search algorithm
search:
	$(GCC) src/algo_search.c src/algo_search_gui.c src/algo_search_stat.c src/utilities/interactions.c

# obstacle avoidance with spp terms
obstacle:
	$(GCC) src/algo_obstacle_avoidance_spp.c src/algo_virtualtarg_gui.c src/algo_spp_stat.c

# demo model for marketing activity
demo:
	$(GCC) src/algo_demo.c src/algo_demo_gui.c src/algo_demo_stat.c

# snake model for collective waypoint following
snake:
	$(GCC) src/algo_snake.c src/algo_snake_gui.c src/algo_snake_stat.c

# Removing binaries
clean:
	rm -rf robotflocksim_main
