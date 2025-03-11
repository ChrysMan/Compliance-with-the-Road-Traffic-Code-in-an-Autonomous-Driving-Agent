/*
 * File:         crossingObstacle.c
 * Description:  This is an empty robot controller, the robot does nothing.
 */

#include <webots/robot.h>
#include <webots/supervisor.h>

  //our human walks from y = 37 to y = 53.5
double maxY = 53.5f;
double minY = 37.0f;
double mooveStep = 0.012f;
bool moovingUp = true;
  
int main() {
  wb_robot_init();
  int time_step = wb_robot_get_basic_time_step();
  if (time_step == 0)
    time_step = 1;
  
  WbNodeRef myself = wb_supervisor_node_get_self();
  double newVals[3];
  for (;;){
  
    WbFieldRef objPos = wb_supervisor_node_get_field(myself, "translation");
    const double *values = wb_supervisor_field_get_sf_vec3f(objPos);
    newVals[0] = values[0];
    newVals[1] = values[1];
    newVals[2] = values[2];
    double curY = values[1];
    if(moovingUp){
      newVals[1] = curY + mooveStep;
      wb_supervisor_field_set_sf_vec3f(objPos, newVals);
      
      if(curY >= maxY - mooveStep){
        moovingUp = false;
      }
    }
    else{
      newVals[1] = curY - mooveStep;
      wb_supervisor_field_set_sf_vec3f(objPos, newVals);
      
      if(curY <= minY + mooveStep){
        moovingUp = true;
      }
    }
    wb_robot_step(time_step);
  }
    
  return 0;
}
