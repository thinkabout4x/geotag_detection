import cv2
import numpy as np
import os
import sys


def nothing(*arg):
    pass


def switch(*arg):
    pass


def ratio_calc(value_1, value_2):
    if value_1 >= value_2:
        ratio = value_1/value_2
    else:
        ratio = value_2/value_1
    return ratio


def geotag_finder(path_to_picture, ratio_square=1.2, ratio_area=2, settings=False):
    stop_flag = True
    success = True
    red_area = 0
    white_area = 0
    resize_coeff = 0.3

    h_r_l = 160
    s_r_l = 25  # 75
    v_r_l = 105  # 105
    h_r_h = 180
    s_r_h = 255
    v_r_h = 255

    h_w_l = 17
    s_w_l = 0
    v_w_l = 225
    h_w_h = 180
    s_w_h = 50
    v_w_h = 255

    img = cv2.imread(path_to_picture)
    if img is not None:
        height, width, channels = img.shape
    else:
        return None
    if settings:
        cv2.namedWindow("settings")
        # red
        cv2.createTrackbar('h_r_l', 'settings', h_r_l, 180, nothing)
        cv2.createTrackbar('s_r_l', 'settings', s_r_l, 255, nothing)
        cv2.createTrackbar('v_r_l', 'settings', v_r_l, 255, nothing)
        cv2.createTrackbar('h_r_h', 'settings', h_r_h, 180, nothing)
        cv2.createTrackbar('s_r_h', 'settings', s_r_h, 255, nothing)
        cv2.createTrackbar('v_r_h', 'settings', v_r_h, 255, nothing)
        # white
        cv2.createTrackbar('h_w_l', 'settings', h_w_l, 180, nothing)
        cv2.createTrackbar('s_w_l', 'settings', s_w_l, 255, nothing)
        cv2.createTrackbar('v_w_l', 'settings', v_w_l, 255, nothing)
        cv2.createTrackbar('h_w_h', 'settings', h_w_h, 180, nothing)
        cv2.createTrackbar('s_w_h', 'settings', s_w_h, 255, nothing)
        cv2.createTrackbar('v_w_h', 'settings', v_w_h, 255, nothing)
        # start/stop
        cv2.createTrackbar('stop', 'settings', 0, 1, nothing)

    while True:
        if settings:
            h_r_l = cv2.getTrackbarPos('h_r_l', 'settings')
            s_r_l = cv2.getTrackbarPos('s_r_l', 'settings')
            v_r_l = cv2.getTrackbarPos('v_r_l', 'settings')
            h_r_h = cv2.getTrackbarPos('h_r_h', 'settings')
            s_r_h = cv2.getTrackbarPos('s_r_h', 'settings')
            v_r_h = cv2.getTrackbarPos('v_r_h', 'settings')

            h_w_l = cv2.getTrackbarPos('h_w_l', 'settings')
            s_w_l = cv2.getTrackbarPos('s_w_l', 'settings')
            v_w_l = cv2.getTrackbarPos('v_w_l', 'settings')
            h_w_h = cv2.getTrackbarPos('h_w_h', 'settings')
            s_w_h = cv2.getTrackbarPos('s_w_h', 'settings')
            v_w_h = cv2.getTrackbarPos('v_w_h', 'settings')
            stop_flag = bool(cv2.getTrackbarPos('stop', 'settings'))

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(img_hsv, (h_r_l, s_r_l, v_r_l), (h_r_h, s_r_h, v_r_h))
        mask_white = cv2.inRange(img_hsv, (h_w_l, s_w_l, v_w_l), (h_w_h, s_w_h, v_w_h))
        mask_or = cv2.bitwise_or(mask_white, mask_red)
        masked = cv2.bilateralFilter(mask_or, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if settings:
            img_resize = cv2.resize(img,
                                    (int(width * resize_coeff), int(height * resize_coeff)))
            masked_resize = cv2.resize(masked, (
                int(width * resize_coeff), int(height * resize_coeff)))
            thresh_resize = cv2.resize(thresh, (
                int(width * resize_coeff), int(height * resize_coeff)))
            cv2.imshow('region', img_resize)
            cv2.imshow('mask', masked_resize)
            cv2.imshow('threshold', thresh_resize)

        if contours:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if ratio_calc(w, h) <= ratio_square:
                    cand_roi = img[y:y + h, x:x + w]
                    cand_roi_hsv = cv2.cvtColor(cand_roi, cv2.COLOR_BGR2HSV)
                    cand_roi_red = cv2.inRange(cand_roi_hsv, (h_r_l, s_r_l, v_r_l), (h_r_h, s_r_h, v_r_h))
                    cand_roi_white = cv2.inRange(cand_roi_hsv, (h_w_l, s_w_l, v_w_l), (h_w_h, s_w_h, v_w_h))
                    thresh_red = cv2.adaptiveThreshold(cand_roi_red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 11, 2)
                    thresh_white = cv2.adaptiveThreshold(cand_roi_white, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY_INV, 11, 2)
                    contour_red, _ = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contour_white, _ = cv2.findContours(thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt_red in contour_red:
                        red_area += cv2.contourArea(cnt_red)
                    for cnt_white in contour_white:
                        white_area += cv2.contourArea(cnt_white)

                    if red_area != 0 and white_area != 0 and ratio_calc(red_area, white_area) <= ratio_area:
                        cand_roi = cv2.GaussianBlur(cand_roi, (5, 5), 0)
                        cand_roi = cv2.cvtColor(cand_roi, cv2.COLOR_BGR2GRAY)
                        _, cand_roi_thresh = cv2.threshold(cand_roi, 200, 255, 0)
                        contour_cand, _ = cv2.findContours(cand_roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if contour_cand:
                            for cnt_cand in contour_cand:
                                epsilon = 0.05 * cv2.arcLength(cnt_cand, True)
                                approx = cv2.approxPolyDP(cnt_cand, epsilon, True)
                                if len(approx) == 6 and not cv2.isContourConvex(cnt_cand):  # 6-i grannik vognutiy
                                    cand_roi_thresh = cv2.cvtColor(cand_roi_thresh, cv2.COLOR_GRAY2BGR)
                                    cv2.drawContours(cand_roi_thresh, [approx], -1, (0, 0, 255), 2)
                                    cv2.drawContours(cand_roi_thresh, contour_cand, -1, (0, 255, 0), 1)
                                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    x_coord = int(x + w / 2)
                                    y_coord = int(y + h / 2)
                                    cv2.line(img, (0, y_coord), (x_coord, y_coord), (255, 0, 0), 3)
                                    cv2.line(img, (x_coord, 0), (x_coord, y_coord), (0, 0, 255), 3)
                                    coords = (x_coord, y_coord)
                                    if settings:
                                        cv2.imshow('cand', cand_roi_thresh)
                                    else:
                                        return img, success, coords
                    red_area = 0
                    white_area = 0
        cv2.waitKey(20)
        if stop_flag and settings:
            cv2.destroyAllWindows()
            return None
        elif not settings:
            return img, not success


if __name__ == "__main__":
    # cwd = os.getcwd()
    # directory = os.path.join(cwd, "result")
    # if not os.path.isdir(directory):
    #     os.makedirs(directory)
    #
    # path = os.path.join(os.getcwd(), "Photos")
    # photos = os.listdir(path)
    # i = 0
    # for pht in photos:
    #     image_and_coordinates = geotag_finder(os.path.join(path, pht))
    #     if image_and_coordinates is not None:
    #         #cv2.imshow('region', image_and_coordinates[0])
    #         os.chdir(directory)
    #         if image_and_coordinates[1]:
    #             cort = image_and_coordinates[2]
    #             name = str(i)+'_'+str(int(image_and_coordinates[1]))+'_'+str(cort[0])+'_'+str(cort[1])+".JPEG"
    #             print(name)
    #             cv2.imwrite(name, image_and_coordinates[0])
    #             i += 1
    #         else:
    #             name = str(i)+'_'+str(int(image_and_coordinates[1]))+".JPEG"
    #             print(name)
    #             cv2.imwrite(name, image_and_coordinates[0])
    #             i += 1
    #         os.chdir(path)
    #
    #     else:
    #         print('nothing')
    path = os.path.join(os.getcwd(), "test3.JPEG")
    image_and_coordinates = geotag_finder(path, settings=True)


