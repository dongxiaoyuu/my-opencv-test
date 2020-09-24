# PROJECT: Credit card number identification信用卡数字识别，同样的可以做车牌识别

import cv2
import numpy as np
import argparse
import Utils
from imutils import contours


def show_image(winname, mat):
    cv2.imshow(winname, mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_template(template_path):     # 读模板
    template = cv2.imread(template_path)  # read
    show_image('template', template)

    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # gray
    show_image('gray template', gray_template)

    threshold_template = cv2.threshold(gray_template, 10, 255, cv2.THRESH_BINARY_INV)[1]  # binary thresh二值化
    show_image('threshold template', threshold_template)

    return template, threshold_template


def get_template_numbers(template, threshold_template):  # 检测模板外轮廓
    # find all number outer counter

    # contour检测外轮廓，cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
    contours, hierarchy = cv2.findContours(threshold_template.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(template, contours, -1, (0, 0, 255), 3)    #画出轮廓，-1表示画所有的轮廓，

    print(np.array(contours, dtype=object).shape)  # must is 10，10个轮廓
    show_image('new template', template)

    # sort contour
    contours = Utils.sort_contours(contours, method='left-to-right')[0]  # sort对轮廓进行排序，从左到右从上到下

    # save all template numbers in dict
    digits_template_dict = {}

    # 遍历每一个轮廓
    for (i, contour) in enumerate(contours):
        # 计算外接矩形并resize成合适大小
        x, y, w, h = cv2.boundingRect(contour)

        roi = threshold_template[y:y + h, x:x + w]
        roi = cv2.resize(roi, (60, 90))
        # store to dictionary，每个数字对应一个模板
        digits_template_dict[i] = roi

    return digits_template_dict  # 至此模板就处理好了


def read_credit_card(credit_card_path):  # 图像预处理
    credit_card = cv2.imread(credit_card_path)  # read
    show_image('credit_card', credit_card)

    credit_card = Utils.resize(credit_card, width=300)  # resize

    gray_credit_card = cv2.cvtColor(credit_card, cv2.COLOR_BGR2GRAY)  # gray
    show_image('gray_credit_card', gray_credit_card)

    return credit_card, gray_credit_card


def operation_credit_card(gray_credit_card):
    """

    :param gray_credit_card:
    :return:
    """
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # 自己设定一个核
    tophat = cv2.morphologyEx(gray_credit_card, cv2.MORPH_TOPHAT, rect_kernel)  # TOPHAT顶帽操作突出更明亮的区域
    show_image('tophat', tophat)

    # `ksize = #FILTER_SCHARR (-1)` that corresponds to the \f$3\times3\f$ Scharr
    sobelX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # sobel

    sobelX = np.absolute(sobelX)
    (minVal, maxVal) = (np.min(sobelX), np.max(sobelX))
    sobelX = (255 * ((sobelX - minVal) / (maxVal - minVal)))
    sobelX = sobelX.astype('uint8')

    print(np.array(sobelX).shape)
    show_image('sobelX', sobelX)
    # 闭操作，先膨胀后腐蚀将数字连在一起
    sobelX = cv2.morphologyEx(sobelX, cv2.MORPH_CLOSE, rect_kernel)  # CLOSE
    show_image('sobelX', sobelX)

    # 二值化，找出需要的部分，OTSU thresh: auto find threshold value, best in double peak, need set thresh = 0
    thresh = cv2.threshold(sobelX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
        1]  # binary thresh or OTSU thresh (auto find)
    show_image('thresh', thresh)
    #  再来一个闭操作，填补数字的空白
    five_rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close_credit_card = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, five_rect_kernel)  # CLOSE
    show_image('close_credit_card', close_credit_card)

    return close_credit_card


def get_credit_card_counters(operation_result_credit, credit_card):
    # 计算、画出信用卡数字的轮廓
    contours, hierarchy = cv2.findContours(operation_result_credit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    conts = contours
    credit_card_copy = credit_card.copy()
    cv2.drawContours(credit_card_copy, conts, -1, (0, 0, 255), 3)  # draw contours
    show_image('credit_card_copy', credit_card_copy)

    # get all number string array
    number_string = []
    # 遍历当前的轮廓
    for (i, c) in enumerate(conts):
        #  计算矩形
        (x, y, w, h) = cv2.boundingRect(c)

        ar = w / float(h)  # weight / height = often number string percentage

        # 选择合适的区域，根据实际任务来，这里的数字都是4个一组
        if ar > 2.5 and ar < 4.0:  # 通过判断把符合的留下来
            if (w > 40 and w < 55) and (h > 10 and h < 20):
                number_string.append((x, y, w, h))
    #  把符合的轮廓从左到右排序
    number_string = sorted(number_string, key=lambda x: x[0])  # sorted by x direction location
    return number_string


def get_credit_card_numbers(credit_card, gray_credit_card, credit_card_number_counters, digits_template_dict):
    result_numbers = []

    #  遍历每一个大轮廓中的数字（小轮廓）
    for (i, (gx, gy, gw, gh)) in enumerate(credit_card_number_counters):
        group = gray_credit_card[gy - 5:gy + gh + 5, gx - 5: gx + gw + 5]   # extend the number group edge定位轮廓，根据坐标提取每一个组，
        show_image('group', group)  # 输出一个大轮廓
        # 对输出的轮廓处理
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # threshold
        show_image('group', group)

        credit_card_contours, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_SIMPLE)  # contour

        credit_card_contours = contours.sort_contours(credit_card_contours, method='left-to-right')[0]

        result_group = []
        # 计算每一组中的每一个数值
        for c in credit_card_contours:
            # 找到当前数字的轮廓，resize成合适大小
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (60, 90))  # 做模板匹配大小都得是一样的
            show_image('roi', roi)

            # template result dependency score计算匹配得分
            scores = []
            for (digit, digit_roi) in digits_template_dict.items():
                # 模板匹配
                result = cv2.matchTemplate(roi, digit_roi, cv2.TM_CCOEFF)  # use CCOEFF, get max value
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            result_group.append(str(np.argmax(scores)))  # add max value找到最合适的数字

        # draw number counter
        cv2.rectangle(credit_card, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
        # put result value in the credit card
        cv2.putText(credit_card, "".join(result_group), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        result_numbers.extend(result_group)

    return result_numbers


if __name__ == '__main__':
    # Edit configurations
    # --image ./CreditCardData/credit_card_01.png --template ./CreditCardData/ocr_a_reference.png
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, help='Credit card picture path')
    parser.add_argument('-t', '--template', required=True, help='Template numbers picture path')
    args = vars(parser.parse_args())

    # read template
    template, threshold_template = read_template(args.get('template') or args.get('i'))

    # all template digits
    digits_template_dict = get_template_numbers(template, threshold_template)

    # read credit card
    credit_card, gray_credit_card = read_credit_card(args.get('image') or args.get('t'))

    # some picture operation
    operation_result_credit = operation_credit_card(gray_credit_card)

    # get credit card counters
    credit_card_number_counters = get_credit_card_counters(operation_result_credit, credit_card)

    # use template to get the credit numbers
    result_numbers = get_credit_card_numbers(credit_card, gray_credit_card, credit_card_number_counters,
                                             digits_template_dict)

    print("credit card result number is : {}".format("".join(result_numbers)))
    show_image('Credit card', credit_card)
