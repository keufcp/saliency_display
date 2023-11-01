import cv2
import utils.saliency_detect as saliency_detect

img = cv2.imread("150.jpg")
cv2.imshow("Image", saliency_detect.saliency_detect(img, "SR"))
cv2.waitKey(0)  # 0を指定するとキーが押されるまで待ちます
cv2.destroyAllWindows()  # ウィンドウを閉じます