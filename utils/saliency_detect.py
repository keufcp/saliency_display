import cv2
from config import threshold, alpha

def saliency_detect(img, method="SR"):

    # アルゴリズムの設定
    saliency = None
    if method == 'SR':
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    elif method == 'FG':
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    if saliency is None:
        exit()

    # サリエンシーディテクション
    bool, map = saliency.computeSaliency(img)
    i_saliency = (map * 255).astype("uint8")

    # スレッショルド作成
    if threshold:
        i_threshold = cv2.threshold(i_saliency, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # heatmap作成
    heatmap = cv2.applyColorMap(i_saliency, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

    if threshold:
        combined_image = cv2.hconcat([img, cv2.cvtColor(i_threshold, cv2.COLOR_GRAY2BGR), heatmap])
    else:
        combined_image = cv2.hconcat([img, heatmap])

    return combined_image

    # 画像を保存
    # cv2.imwrite(output_img, heatmap)
    # cv2.imwrite("th_" + output_img, i_threshold)

# cv2.imshow("Image", saliency_detect("150.jpg", "SR"))
# cv2.waitKey(0)  # 0を指定するとキーが押されるまで待ちます
# cv2.destroyAllWindows()  # ウィンドウを閉じます