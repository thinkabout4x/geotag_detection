import math

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
        ratio = (value_1-value_2)/value_1
    else:
        ratio = (value_2-value_1)/value_2
    return ratio


def resize_image(image, w, h):
    height, width = image.shape[:2]
    res = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    return res

def biggest_contour(contours):
    result_area = 0
    output = None
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area >= result_area:
            result_area = cnt_area
            output = cnt
    return output


def sum_of_contouts_area(contours):
    area = 0
    for cnt in contours:
        area += cv2.contourArea(cnt)
    return area


def warp_affine_without_crop(img, angle):
    (h, w) = img.shape
    img_center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(img_center, angle, 1)
    abs_cos = abs(m[0, 0])
    abs_sin = abs(m[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    m[0, 2] += bound_w/2 - img_center[0]
    m[1, 2] += bound_h/2 - img_center[1]

    img_rot = cv2.warpAffine(img, m, (bound_w, bound_h))
    return img_rot


def img_crop(img, pts):
    xs = [i[0] for i in pts]
    ys = [i[1] for i in pts]
    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)
    img = img[y1:y2, x1:x2]
    return img


def geotag_finder(path_to_picture, ratio_square=0.5, ratio_area=0.5, settings=False):
    stop_flag = True
    success = True
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
                    red_area = sum_of_contouts_area(contour_red)
                    white_area = sum_of_contouts_area(contour_white)
                    if red_area != 0 and white_area != 0 and ratio_calc(red_area, white_area) <= ratio_area:
                        cand_roi = cv2.cvtColor(cand_roi, cv2.COLOR_BGR2GRAY)
                        _, cand_roi_thresh = cv2.threshold(cand_roi, 200, 255, 0)
                        contour_cand, _ = cv2.findContours(cand_roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        if contour_cand:
                            cnt_cand = biggest_contour(contour_cand)
                            cnt_cand_rect = cv2.minAreaRect(cnt_cand)
                            rect_box = np.int0(cv2.boxPoints(cnt_cand_rect))
                            if cv2.arcLength(cnt_cand, True) >= cv2.arcLength(rect_box, True):
                                cnt_cand_rect_angle = cnt_cand_rect[2]
                                cand_roi_thresh_rotated = warp_affine_without_crop(cand_roi_thresh, cnt_cand_rect_angle)
                                _, cand_roi_thresh_rotated = cv2.threshold(cand_roi_thresh_rotated, 200, 255, 0)
                                contour_cand_rot, _ = cv2.findContours(cand_roi_thresh_rotated, cv2.RETR_TREE,
                                                                       cv2.CHAIN_APPROX_SIMPLE)
                                if contour_cand_rot:
                                    cnt_cand_rot = biggest_contour(contour_cand_rot)
                                    cnt_cand_rect_rot = cv2.minAreaRect(cnt_cand_rot)
                                    rect_box_rot = np.int0(cv2.boxPoints(cnt_cand_rect_rot))
                                    cnt_cand_rect_rot_w = int(cnt_cand_rect_rot[1][0])
                                    cnt_cand_rect_rot_h = int(cnt_cand_rect_rot[1][1])
                                    cand_roi_thresh_rotated = img_crop(cand_roi_thresh_rotated, rect_box_rot)
                                    if cand_roi_thresh_rotated is not None:
                                        if cnt_cand_rect_rot_w >= 7 and cnt_cand_rect_rot_h >= 7:
                                            cand_roi_thresh_rotated = resize_image(cand_roi_thresh_rotated, 100, 100)
                                            (mask_h, mask_w) = cand_roi_thresh_rotated.shape
                                            _, cand_roi_thresh_rotated = cv2.threshold(cand_roi_thresh_rotated, 200,
                                                                                       255, 0)
                                            cv2.imshow('cand_rot_test', cand_roi_thresh_rotated)
                                            while True:
                                                key = cv2.waitKey(20)
                                                if key == ord('n'):
                                                    cv2.destroyWindow('cand_rot_test')
                                                    break
                                            cand_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
                                            mask_poly_1 = np.array([[0, 0], [mask_h, 0], [mask_w/2, mask_h/2]],dtype=np.int32)
                                            mask_poly_2 = np.array([[0, mask_h], [mask_w, mask_h], [mask_w/2, mask_h/2]],dtype=np.int32)
                                            cv2.fillPoly(cand_mask, [mask_poly_1], (255, 255, 255))
                                            cv2.fillPoly(cand_mask, [mask_poly_2], (255, 255, 255))
                                            cand_mask_inv = cv2.bitwise_not(cand_mask)
                                            xor = cv2.bitwise_xor(cand_roi_thresh_rotated, cand_mask)
                                            xor_inv = cv2.bitwise_xor(cand_roi_thresh_rotated, cand_mask_inv)
                                            cand_mask_contours_area = cv2.countNonZero(cand_mask)
                                            xor_area = cv2.countNonZero(xor)
                                            xor_inv_area = cv2.countNonZero(xor_inv)
                                            ratio = ratio_calc(cand_mask_contours_area, xor_area)
                                            ratio_inv = ratio_calc(cand_mask_contours_area, xor_inv_area)
                                            print(ratio)
                                            print(ratio_inv)
                                            print(cand_mask_contours_area)
                                            print(xor_area)
                                            print(xor_inv_area)
                                            if (cand_mask_contours_area > xor_inv_area and ratio_inv >= 0.5 and ratio < 0.6) or (cand_mask_contours_area > xor_area and ratio >= 0.5 and ratio_inv < 0.6):
                                                xor = cv2.cvtColor(xor, cv2.COLOR_GRAY2BGR)
                                                cand_roi_thresh = cv2.cvtColor(cand_roi_thresh, cv2.COLOR_GRAY2BGR)
                                                cv2.drawContours(cand_roi_thresh, cnt_cand_rot, -1, (0, 255, 0), 1)
                                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                                x_coord = int(x + w / 2)
                                                y_coord = int(y + h / 2)
                                                cv2.line(img, (0, y_coord), (x_coord, y_coord), (255, 0, 0), 3)
                                                cv2.line(img, (x_coord, 0), (x_coord, y_coord), (0, 0, 255), 3)
                                                coords = (x_coord, y_coord)
                                                if settings:
                                                    cv2.imshow('cand_rot', cand_roi_thresh_rotated)
                                                    cv2.imshow('cand_mask', cand_mask)
                                                    cv2.imshow('xor', xor)
                                                    cv2.imshow('xor_inv', xor_inv)
                                                else:
                                                    return img, success, coords
        cv2.waitKey(20)
        if stop_flag and settings:
            cv2.destroyAllWindows()
            return None
        elif not settings:
            return img, not success


if __name__ == "__main__":
    cwd = os.getcwd()
    directory = os.path.join(cwd, "result")
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # path = os.path.join(os.getcwd(), "Photos_spring")
    # photos = os.listdir(path)
    # i = 0
    # for pht in photos:
    #     image_and_coordinates = geotag_finder(os.path.join(path, pht))
    #     if image_and_coordinates is not None:
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
    path = os.path.join(os.getcwd(), "test11.JPEG")
    image_and_coordinates = geotag_finder(path, settings=True)


