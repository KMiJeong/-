import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui as pag

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [ 362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398 ]
RIGHT_EYE = [ 33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246 ]

LEFT_IRIS = [ 474,475,476,477 ]
RIGHT_IRIS = [ 469,470,471,472 ]

with mp_face_mesh.FaceMesh(max_num_faces = 1,
                           refine_landmarks = True,
                           min_detection_confidence = 0.5,
                           min_tracking_confidence = 0.5
) as face_mesh:

    capture = cv.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])
            
            left_eye_pts = mesh_points[LEFT_EYE]
            right_eye_pts = mesh_points[RIGHT_EYE]

            (l_cx,l_cy), l_rad = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy), r_rad = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])

            l_center_left = np.array([l_cx,l_cy], dtype=np.int32)
            r_center_right = np.array([r_cx,r_cy], dtype=np.int32)

            # calculate the relative position of the iris centers
            rel_pos = r_center_right - l_center_left

            # calculate the amount of movement based on the relative position of the iris centers
            move_x = int(20 * rel_pos[0] / img_w)
            move_y = int(15 * rel_pos[1] / img_h)

            # move the cursor based on the amount of movement calculated
            pag.moveRel(move_x, move_y)

        cv.imshow('main', frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
