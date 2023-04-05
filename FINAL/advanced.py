import cv2
import numpy as np
import math
import ppadb
import uiautomator2.exceptions
#from uiautomator2 import PointerGesture
from ppadb.client import Client as AdbClient
import threading
import uiautomator2 as u2
import time
from datetime import datetime, timedelta

print(uiautomator2.version)

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
    bottomRight = [415,80]
    bottomLeft = [233,93]
    topRight = [471,401]
    topLeft = [241,403]
    orig_img_coor = np.float32([bottomLeft, bottomRight, topLeft, topRight])
    height = 2244
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


def findDistance(element1, element2):
    xdiff = element2[0] - element1[0]
    ydiff = element2[1] - element2[1]

    return (xdiff ** 2 + ydiff ** 2) ** 0.5

def reduceThresholdofPoints(element1, element2, threshold):
    if(findDistance(element1, element2) <= threshold):

        return -2

    else:
        return findDistance(element1, element2)

##########################################################################
# Obtaining the reference image

input_location_list = []
send_input_list = []

input_location_prev = [-1, -1, -1]

finger_down_location = [-1, -1, -1]
finger_up_location = [-1, -1, -1]


error_release_status = 0
hold_release_status = 0

time_now = 0
time_prev = 0
untrigger_counter = 0
untrigger_counter_thresh = 3

time_error_now = 0
time_error_prev = 0
#delta_time_error = 0

hold_frame_counter = 0

next_trigger_interval_microsec = 300
action = ""

release_hold_frame_counter = 0

lock = threading.Lock()
time_lock = threading.Lock()
sem = threading.Semaphore(0)
sem_time_prevent = threading.Semaphore(1)


def fingerDetection():

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
    threshold_value = 60


    # Blob detection to avoid 2 or more blobs

    # Create a blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 1000000
    detector = cv2.SimpleBlobDetector_create(params)

    slide_in_out = False
    # Applying the image processing, thresholding, and finding contours and hulls
    while True:
        global input_location_prev
        global time_now
        global time_prev

        global time_error_now
        global time_error_prev

        global untrigger_counter
        # global delta_time_error

        global error_release_status
        global hold_frame_counter
        global hold_release_status

        global finger_down_location
        global finger_up_location

        global action
        global release_hold_frame_counter

        input_location = [-1, -1, -1]



        # Read the next frame
        ret, frame2 = cap.read()
        frame2 = cv2.flip(frame2, 1)

        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        keypoints = detector.detect(gray2)
        #print("Number of blobs:", len(keypoints))

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
        kernel_sec = (3,3)

        diff = cv2.GaussianBlur(diff, kernel, 0)

        ret, threshold = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        threshold = cv2.GaussianBlur(threshold, kernel_sec, 0)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

        threshold = cv2.GaussianBlur(threshold, kernel, 0)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_threshold = 5000
        area_threshold2 = 400000

        hulls = []
        pt_list = []
        for cnt in contours:
            contour_area_sum = 0
            hull = cv2.convexHull(cnt)
            hulls.append(hull)
            area = cv2.contourArea(hull)

            if(area > area_threshold):

                M = cv2.moments(hull)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                point = (cx, cy)
                pt_list.append(point)
                cv2.circle(frame2, point, 15, (0, 0, 255), -1)

        if(len(pt_list) == 2):
            intersect = cv2.intersectConvexConvex(hulls[-2], hulls[-1])
            if intersect[0] > 100:
                #print("Number of intersections:", intersect[0])
                #print("Hull", -2, "and hull", -1, "are overlapping.")
                slide_in_out = False
            else:
                slide_in_out = True

        elif(len(pt_list) == 1):
            if(pt_list[0][1] < 45):
                slide_in_out = True
            else:
                slide_in_out = False

        else:
            slide_in_out = True

        if(slide_in_out == False):
            for cnt in contours:
                hulls.append(hull)
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


                    #################################### Noise cancellation and increase accuracy
                    sum_y_offset = abs(highest_points[0][0][1] - highest_points[1][0][1]) + abs(highest_points[1][0][1] - highest_points[2][0][1]) + abs(highest_points[0][0][1] - highest_points[2][0][1])
                    avrg_y_offset = sum_y_offset/3

                    sum_y = abs(highest_points[0][0][1]) + abs(highest_points[0][0][1]) + abs(highest_points[2][0][1])
                    avrg_y = sum_y/3

                    higher_than_20_points = 0

                    # Y offsets are large (not close)
                    if(avrg_y_offset > 50):
                        for pt in highest_points:
                            if(tuple(pt[0])[1] >= int(avrg_y)):
                                higher_than_20_points+=1

                        # If only 1 point has an y larger than the referecne
                        if(higher_than_20_points <= 1):
                            decision_point = sorted_pts[-1]
                            cv2.circle(frame2, decision_point[0], 10, (0, 255, 255), 0)

                        # Else if 2 points have y larger than the reference
                        else:
                            if(abs(sorted_pts[-2][0][1] - avrg_y) < abs(sorted_pts[-2][0][1] - sorted_pts[-1][0][1])):
                                decision_point = sorted_pts[-1]
                                cv2.circle(frame2, decision_point[0], 10, (0, 255, 255), 0)

                            elif(sorted_pts[-2][0][1]  - sorted_pts[-2][0][1] >= 20):
                                decision_point = sorted_pts[-1]
                                cv2.circle(frame2, decision_point[0], 10, (0, 255, 255), 0)
                            else:
                                decision_point = sorted_pts[-2:]
                                for pt in decision_point:
                                    cv2.circle(frame2, tuple(pt[0]), 10, (0, 255, 255), 0)

                    # Three points y values are close
                    else:
                        # 3 points are very flat
                        if(findSlope(highest_points[0], highest_points[1]) * findSlope(highest_points[0], highest_points[1]) * findSlope(highest_points[0], highest_points[1]) <= 0.25):

                            # Point 0 and point 1 has the biggest x offsets
                            if(abs(highest_points[0][0][0] - highest_points[1][0][0]) >= abs(highest_points[1][0][0] - highest_points[2][0][0]) \
                                    and abs(highest_points[0][0][0] - highest_points[1][0][0]) >= abs(highest_points[0][0][0] - highest_points[2][0][0])):

                                decision_point = [highest_points[0], highest_points[1]]

                            # Point 1 and 2 has the biggest X offsets
                            elif(abs(highest_points[1][0][0] - highest_points[2][0][0]) >= abs(highest_points[0][0][0] - highest_points[2][0][0]) \
                                 and abs(highest_points[1][0][0] - highest_points[2][0][0]) >= abs(highest_points[0][0][0] - highest_points[1][0][0])):
                                #compare highest point with avg of two side point
                                decision_point1 = [int(0.5*(highest_points[1][0][0] + highest_points[2][0][0])), int(0.5*(highest_points[1][0][1] + highest_points[2][0][1]))]
                                decision_point = [highest_points[0], decision_point1]

                            # Point
                            else:
                                decision_point = [highest_points[0], highest_points[1]]

                            try:
                                for pt in decision_point:
                                    cv2.circle(frame2, tuple(pt[0]), 10, (0, 255, 255), 0)
                                    #print("Passed.", pt)
                            except TypeError:
                                print("Temporary Error", decision_point)

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



                    ##################### Detect finger triggering and status (including in, triggering, out)
                    trigger_thresh_upper = 35
                    trigger_thresh_lower = 25


                    if(len(decision_point) == 2):
                        try:
                            x_offset = abs(decision_point[-1][0][0] - decision_point[-2][0][0])
                        except TypeError:
                            x_offset = abs(decision_point[-1][0] - decision_point[-2][0][0])

                        # 2 Points are detected, so and the distance between these 2 points are within detection range
                        if(x_offset >= trigger_thresh_lower and x_offset <= trigger_thresh_upper):
                            try:
                                triggered_location = [int((0.5*(decision_point[0][0][0] + decision_point[1][0][0]))), int(0.5*(decision_point[0][0][1] + decision_point[1][0][1]))]

                            except TypeError:
                                print("Error on detection")
                            #print("Triggerd Location:", triggered_location)

                        else:
                            triggered_location = [-1, -1, -1]
                    else: # One point is found in opencv, meaning that point is the triggered point
                        triggered_location = [int(decision_point[0][0]), int(decision_point[0][1])]

                        #print("Time elapsed:", untrigger_counter)
                        #print("Triggerd Location:", triggered_location)


                    # Remove any detection errors between frames (if the finger detection has lost for
                    # frames within detection threshold time)
                    if(triggered_location[0] != -1):
                        cv2.circle(frame2, tuple(triggered_location), 10, (0, 255, 255), 0)
                        text3 = f"Point on Trigger: ({triggered_location[0]},{triggered_location[1]})"
                        cv2.putText(frame2, text3, (10, 90), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        output_x, output_y = finger_coordinate_to_adb_coordinates(int(triggered_location[0]), int(triggered_location[1]) - 10)
                        lock.acquire()
                        if(reduceThresholdofPoints([output_x, output_y], input_location_prev, 60) != -2):
                            # time_now = time.time()
                            # untrigger_counter = time_now - time_prev
                            input_location = [output_x, output_y]
                            #print("Time elapsed:", untrigger_counter)

                        else:
                            input_location = input_location_prev
                        lock.release()


                else: # update the new frame if there are small changes to the background
                    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Detect if the finger action is swipe, touch, or hold
        lock.acquire()
        error_pixel_distance_1 = 50
        error_pixel_distance_2 = 100



        if(input_location[0] == -1):
            # case 1: startup
            #TODO: Revise the startup of the send input list
            if(untrigger_counter == 0):
                #time_error_now = time.time()
                #print("time error now ", time_error_now)
                #error_release_status = 0
                hold_release_status = 0
                send_input_list.append(input_location)

            else:
                # case 2: release but the time period between 2 trigger is less than 8 frames, and previous triggered, meaning it is holding?
                # Noise cancellation
                if(untrigger_counter <= untrigger_counter_thresh): #TODO: the threshold time here
                    input_location = input_location_prev
                    #print("Error here:", input_location_prev)
                    hold_release_status = 1 #Remove the noises of -1 in the detection


                # It is a trigger reset, processing normally
                else:
                    release_hold_frame_counter = hold_frame_counter # used for detect if its a click

                    if(release_hold_frame_counter <= 8 and release_hold_frame_counter > 0):
                        action = "Click"
                    hold_release_status = 0
                    hold_frame_counter = 0; #The trigger is a reset

                    # Record the finger up location used for control
                    finger_up_location = input_location_prev
                    finger_down_location = [-1, -1, -1]





                send_input_list.append(input_location)

            # Add frames for the untriggered frames
            untrigger_counter += 1


        else: # Triggered, input location[0] != -1
            # Case 3: From triggered to not triggered, count the number of frames that finger is holding on, delta time error: time of -1 detected
            #if(hold_release_status == 1):
            hold_frame_counter += 1
            release_hold_frame_counter = 0

            if(finger_down_location[0] == -1):
                finger_down_location = input_location
                finger_up_location = [-1, -1, -1]

            send_input_list.append(input_location)

            # Reset the -1 counters
            untrigger_counter = 0
            #hold_frame_counter = 0
        #print("Hold frame counter = ", hold_frame_counter)
        #print("hold")
        #sem.release()

        if(len(send_input_list) == 1):
            #print(send_input_list)
            #print(input_location_list)
            sem.release()

        lock.release()
        # Display the thresholded image
        cv2.imshow("Thresholded", threshold)
        cv2.imshow("Diff", diff)
        cv2.imshow("Frame", frame2)


        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def phoneControl():
    global time_now
    global time_prev

    #global time_error_now
    #global time_error_prev

    global input_location_prev
    global finger_up_location
    global finger_down_location

    global untrigger_counter
    global hold_release_status

    global hold_frame_counter
    global release_hold_frame_counter
    global action

    finger_up_bool = False
    finger_down_bool = False

    finger_down_location_prev = [-1,-1,-1]





    while True:
        #print(finger_down_bool)
        d = u2.connect()
        sem.acquire()
        lock.acquire()



        first_element = send_input_list.pop(0)
        #print(first_element)
        lock.release()
        if(first_element[0] != -1):
            input_location_prev = first_element
        lock.acquire()

        if(untrigger_counter > untrigger_counter_thresh + 1):
            input_location_prev = [-1, -1, -1]


        # Click
        click = False
        try:
            if(release_hold_frame_counter < 20 and release_hold_frame_counter > 0):
                click = True
                print("Clicked")

                d.click(finger_up_location[0], finger_up_location[1])
                finger_down_bool = False
                finger_up_location = [-1, -1, -1]
                finger_down_location_prev = [-1,-1,-1]

                click_delay = 0
                while click_delay < 100:
                    click_delay += 1

            else:
                #print("Finger down bool:", finger_down_bool)
                if(finger_down_location[0] != -1 and finger_down_bool == False):
                    # print("Finger down at:", finger_down_location)
                    finger_down_bool = True
                    d.touch.down(finger_down_location[0], finger_down_location[1])
                    finger_down_location_prev = finger_down_location

                if(first_element[0] != -1 and finger_down_bool == True):
                    if(click != True):
                        #print("Finger moving at:", first_element)
                        d.touch.move(first_element[0], first_element[1])


                if(finger_up_location[0] != -1 and finger_down_bool == True):
                    #print("down prev is , up location is ",finger_down_location_prev[0], finger_up_location[0])


                    if(reduceThresholdofPoints(finger_up_location,finger_down_location_prev, 200) != -2):
                        #print("swipe")
                        #print("Finger up location at:", finger_up_location)

                        d.touch.up(finger_up_location[0], finger_up_location[1])

                        finger_down_bool = False
                        finger_up_location = [-1, -1, -1]
                        finger_down_location_prev = [-1,-1,-1]
        except uiautomator2.exceptions.JSONRPCError:
            print("Temprorary hold")



        lock.release()





thread1 = threading.Thread(target=fingerDetection)
thread2 = threading.Thread(target=phoneControl)

thread1.start()
thread2.start()

thread1.join()
thread2.join()
# Release the camera and close all windows
