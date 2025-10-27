#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Secondary Vehicle

import rospy
import numpy as np
from geometry_msgs.msg import Twist, TransformStamped
from math import atan2

class SecondaryVehicleController:
    def __init__(self):
        rospy.init_node('secondary_vehicle_controller', anonymous=True)

        # Subscribe to Vicon topics for primary and secondary vehicles
        self.primary_sub = rospy.Subscriber('/vicon/Limo_04/Limo_04', TransformStamped, self.primary_callback)
        self.secondary_sub = rospy.Subscriber('/vicon/Limo_02/Limo_02', TransformStamped, self.secondary_callback)

        # Publisher for velocity command
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # State
        self.primary_pos = None
        self.secondary_pos = None

        # Parameters
        self.max_speed = 0.8
        self.max_turn = 1
        self.warning_border = 0.35   # Start reacting here
        self.hard_border = 0.5      # Hard safety wall
        self.safe_distance = 1
        
        # Start each cycle with a clean Twist
        self.twist = Twist()

    def primary_callback(self, msg):
        self.primary_pos = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z])

    def secondary_callback(self, msg):
        self.secondary_pos = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z])

        # Ensure both positions are available
        if self.primary_pos is None or self.secondary_pos is None:
            self.twist.linear.x  = 0
            self.twist.angular.z = 0
            self.cmd_pub.publish(self.twist)
            return

        x1, y1 = self.primary_pos[0], self.primary_pos[1]
        x2, y2 = self.secondary_pos[0], self.secondary_pos[1]

        y_offset = y2  # Y of the secondary

        # --- Virtual Boundary Check (±0.5m from center) ---
        if abs(y_offset) > self.hard_border:
            rospy.logwarn("Secondary Limo OUT OF BOUNDS! Y = {:.2f} m".format(y_offset))
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            self.cmd_pub.publish(self.twist)
            return
            
        elif abs(y_offset) > self.warning_border:
            rospy.loginfo("Secondary Limo NEAR BORDER. Applying corrective twist.")  
            self.twist.linear.x = self.max_speed * 0.5
            self.twist.angular.z = -1 if y_offset > 0 else 1
            self.cmd_pub.publish(self.twist)
            return
        
        else:
            rospy.loginfo("Secondary Limo Passing!")
            self.twist.linear.x = self.max_speed
            
            distance = np.linalg.norm(self.primary_pos[:2] - self.secondary_pos[:2])
            
            if distance < self.safe_distance:
                if y1 > 0:
                    angle = atan2(-self.warning_border - y1, x1 - x2) 
                else:
                    angle = atan2(self.warning_border - y1, x1 - x2)
                angle = max(-self.max_turn, min(self.max_turn, angle))
                k = 1.5 # tranforming angle to angular velocity
                angular_velocity = angle * k
                self.twist.angular.z = angular_velocity
    
            else:
                self.twist.angular.z = 0
    
            self.cmd_pub.publish(self.twist)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = SecondaryVehicleController()
        node.run()
    except rospy.ROSInterruptException:
        pass
