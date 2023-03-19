import cv2
import numpy as np
import math
import ppadb
from ppadb.client import Client as AdbClient
import uiautomator2 as u2
import time

#d = u2.connect()
# client = AdbClient(host="127.0.0.1", port=5037) # Default is "127.0.0.1" and 5037
# devices = client.devices()
#
# if len(devices) == 0:
#     print('No devices')
#     quit()
#
# device = devices[0]

#print(f'Connected to {device}')

def finger_coordinate_to_adb_coordinates(x,y):
    bottomRight = [470,115]
    bottomLeft = [303,99]
    topRight = [515,463]
    topLeft = [281,468]
    orig_img_coor = np.float32([bottomLeft, bottomRight, topLeft, topRight])
    height = 2400
    width = 1080
    new_img_coor = np.float32([[0,0],[width,0],[0,height],[width,height]])
    transform_matrix = cv2.getPerspectiveTransform(orig_img_coor, new_img_coor)
    old_coor = [x,y,1]
    old_coor = np.reshape(old_coor,(3,1))
    new_coor_intermediate = np.dot(transform_matrix,old_coor)
    new_coor = [new_coor_intermediate[0]/new_coor_intermediate[2],new_coor_intermediate[1]/new_coor_intermediate[2]]

    adb_x_coordinate = np.round(new_coor[0])
    adb_y_coordinate = np.round(height-new_coor[1])
    adb_x_coordinate = max(10,adb_x_coordinate)
    adb_x_coordinate = min(width - 1,adb_x_coordinate)
    adb_y_coordinate = max(10,adb_y_coordinate)
    adb_y_coordinate = min(height - 1,adb_y_coordinate)

    return int(adb_x_coordinate),int(adb_y_coordinate)

def tap(X, Y):
    device.shell('input touchscreen tap '+str(X)+' '+ str(Y))

#################################################################
#Image processing part
# Varying kernel size of mrophological filter based on the height of the frame
def vary_kernel_size(y):
    if(y < 20):
        return 9
    elif(y < 100):
        return 3
    else:
        return 1

def findSlope(point1, point2):
    y_diff = point1[0][1] - point2[0][1]
    x_diff = point1[0][0] - point2[0][0]

    return abs(y_diff/x_diff)
##########################################################################
# Obtaining the reference image
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

frame_count = 0
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Increment the frame counter
    frame_count += 1

    # Check if we have captured 5 frames
    if frame_count == 30:
        # Save the 5th frame as an image file
        cv2.imwrite("5th_frame.png", frame)
        print("Captured")
        break

# Read the first frame
print("Processing")
reference = cv2.imread('5th_frame.png')
gray1 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

###############################################################################
# Define a threshold value
threshold_value = 9
while True:
    # Read the next frame
    ret, frame2 = cap.read()
    frame2 = cv2.flip(frame2, 1)

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Get 4 sections for different thresholding:
    half_witdh = int((frame.shape[1]+1) / 2)
    half_height = int((frame.shape[0]+1) / 2)

    mask = np.zeros_like(gray2)
    mask[0:half_height, 0:half_witdh] = 2 #9
    mask[half_height:frame.shape[0], 0:half_witdh] = 3 #19
    mask[0:half_height, half_witdh:frame.shape[1]] = 1 #11
    mask[half_height:frame.shape[0], half_witdh:frame.shape[1]] = 4 #13

    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(gray1, gray2)


    diff = diff * 3
    # Add LPF and Morphological filters
    kernel = (5, 5)
    kernel_sec = (5,5)

    diff = cv2.GaussianBlur(diff, kernel, 0)

    ret, threshold = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    threshold = cv2.GaussianBlur(threshold, kernel_sec, 0)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

    threshold = cv2.GaussianBlur(threshold, kernel, 0)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_threshold = 8500
    area_threshold2 = 400000
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)

        if(area > area_threshold):
            epsilon = 0.01 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            cv2.drawContours(frame2, [hull], -1, (0, 255, 0), 1)

            # Sort points by y-coordinate
            sorted_pts = approx[np.argsort(approx[:, 0, 1])]

            # Get 3 points with highest y-values
            highest_points = sorted_pts[-3:]

            #time.sleep(0.2)
            # Draw points
            for pt in highest_points:
                cv2.circle(frame2, tuple(pt[0]), 5, (0, 0, 255), -1)


            # Noise cancellation and increase accuracy
            sum_y_offset = abs(highest_points[0][0][1] - highest_points[1][0][1]) + abs(highest_points[1][0][1] - highest_points[2][0][1]) + abs(highest_points[0][0][1] - highest_points[2][0][1])
            avrg_y_offset = sum_y_offset/3

            sum_y = abs(highest_points[0][0][1]) + abs(highest_points[0][0][1]) + abs(highest_points[2][0][1])
            avrg_y = sum_y/3

            higher_than_20_points = 0

            if(avrg_y_offset > 50):
                for pt in highest_points:
                    if(tuple(pt[0])[1] >= int(avrg_y)):
                        higher_than_20_points+=1

                if(higher_than_20_points <= 1):
                    decision_point = sorted_pts[-1]
                    cv2.circle(frame2, decision_point[0], 10, (0, 255, 255), 0)

                else:
                    if(abs(sorted_pts[-2][0][1] - avrg_y) < abs(sorted_pts[-2][0][1] - sorted_pts[-1][0][1])):
                        decision_point = sorted_pts[-1]
                        cv2.circle(frame2, decision_point[0], 10, (0, 255, 255), 0)
                    else:
                        decision_point = sorted_pts[-2:]
                        for pt in decision_point:
                            cv2.circle(frame2, tuple(pt[0]), 10, (0, 255, 255), 0)
            else:
                if(findSlope(highest_points[0], highest_points[1]) * findSlope(highest_points[0], highest_points[1]) * findSlope(highest_points[0], highest_points[1]) <= 0.25):
                    if(abs(highest_points[0][0][0] - highest_points[1][0][0]) >= abs(highest_points[1][0][0] - highest_points[2][0][0]) \
                            and abs(highest_points[0][0][0] - highest_points[1][0][0]) >= abs(highest_points[0][0][0] - highest_points[2][0][0])):

                        decision_point = [highest_points[0], highest_points[1]]

                    elif(abs(highest_points[1][0][0] - highest_points[2][0][0]) >= abs(highest_points[0][0][0] - highest_points[2][0][0]) \
                         and abs(highest_points[1][0][0] - highest_points[2][0][0]) >= abs(highest_points[0][0][0] - highest_points[1][0][0])):

                        decision_point = [highest_points[1], highest_points[2]]

                    else:
                        decision_point = [highest_points[0], highest_points[2]]

                    for pt in decision_point:
                        cv2.circle(frame2, tuple(pt[0]), 10, (0, 255, 255), 0)

                elif(findSlope(highest_points[0], highest_points[1]) <= 0.5):
                    decision_point = [highest_points[0], highest_points[1]]
                    for pt in decision_point:
                        cv2.circle(frame2, tuple(pt[0]), 10, (0, 255, 255), 0)

                elif(findSlope(highest_points[1], highest_points[2]) <= 0.5):
                    decision_point = [highest_points[1], highest_points[2]]
                    for pt in decision_point:
                        cv2.circle(frame2, tuple(pt[0]), 10, (0, 255, 255), 0)

                else:
                    decision_point = [highest_points[0], highest_points[2]]
                    for pt in decision_point:
                        cv2.circle(frame2, tuple(pt[0]), 10, (0, 255, 255), 0)


            font = cv2.FONT_HERSHEY_SIMPLEX

            # text1 = f"Point 1: ({highest_points[0][0][0]},{highest_points[0][0][1]})"
            # text2 = f"Point 2: ({highest_points[1][0][0]},{highest_points[1][0][1]})"
            # cv2.putText(frame2, text1, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.putText(frame2, text2, (10, 60), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # distance = int(((highest_points[0][0][0] - highest_points[1][0][0])**2 + (highest_points[0][0][1] - highest_points[1][0][1])**2)**0.5)

            # x_offset = highest_points[0][0][0] - highest_points[1][0][0]
            # y_offset = highest_points[0][0][1] - highest_points[1][0][1]

            trigger_thresh_upper = 45
            trigger_thresh_lower = 25

            # y_threshold = 30
            trigger_status_distance = 480

            # Detect finger triggering and status (in, triggering, out)
            if(len(decision_point) == 2):
                x_offset = abs(decision_point[-1][0][0] - decision_point[-2][0][0])
                if(x_offset >= trigger_thresh_lower and x_offset <= trigger_thresh_upper):
                    triggered_location = [int((0.5*(decision_point[0][0][0] + decision_point[1][0][0]))), int(0.5*(decision_point[0][0][1] + decision_point[1][0][1]) - 10)]
                    cv2.circle(frame2, tuple(triggered_location), 10, (0, 255, 255), 0)
                    text3 = f"Point on Trigger: ({triggered_location[0]},{triggered_location[1]})"
                    cv2.putText(frame2, text3, (10, 90), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    output_x, output_y = finger_coordinate_to_adb_coordinates(int(triggered_location[0]), int(triggered_location[1]))
                    print("Triggerd Location:", triggered_location)

            else:
                triggered_location = [int(decision_point[0][0]), int(decision_point[0][1])]
                cv2.circle(frame2, tuple(triggered_location), 10, (0, 255, 255), 0)
                text3 = f"Point on Trigger: ({triggered_location[0]},{triggered_location[1]})"
                cv2.putText(frame2, text3, (10, 90), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                output_x, output_y = finger_coordinate_to_adb_coordinates(int(triggered_location[0]), int(triggered_location[1]))
                print("Triggerd Location:", triggered_location)





            # if(abs(x_offset) <= trigger_thresh_upper and abs(x_offset) >= trigger_thresh_lower):
            #     print("Distance = ", x_offset, "Triggered!")
            #
            #     trigger_status_distance = x_offset
            #
            #     triggered_location = [int((0.5*(highest_points[0][0][0] + highest_points[1][0][0]))), int(0.5*(highest_points[0][0][1] + highest_points[1][0][1]) - 10)]
            #     #print("Triggerd Location:", triggered_location)
            #
            #     cv2.circle(frame2, tuple(triggered_location), 10, (0, 255, 255), 0)
            #     text3 = f"Point on Trigger: ({triggered_location[0]},{triggered_location[1]})"
            #     cv2.putText(frame2, text3, (10, 90), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            #     output_x, output_y = finger_coordinate_to_adb_coordinates(int(triggered_location[0]), int(triggered_location[1]))

                #\device.shell('input touchscreen tap '+str(output_x)+' '+ str(output_y))
                #print("Output:", output_x, output_y)
                # output_x = 450
                # output_y = 1000
                #d.click(output_x, output_y)
                #tap(output_x, output_y)

            # elif(abs(y_offset) >= y_threshold):
            #     print("Triggered!")
            #
            #     trigger_status_distance = 0
            #
            #     triggered_location = [int(highest_points[1][0][0]), int(highest_points[1][0][1]- 10)]
            #     #print("Triggerd Location:", triggered_location)
            #     text3 = f"Point on Trigger: ({triggered_location[0]},{triggered_location[1]})"
            #     cv2.putText(frame2, text3, (10, 90), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # else:
            #     print("Distance = ", x_offset, "Not Triggered!!")

        else:
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the thresholded image
    cv2.imshow("Thresholded", threshold)
    cv2.imshow("Diff", diff)
    cv2.imshow("Frame", frame2)


    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows