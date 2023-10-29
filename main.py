import cv2 as cv
import numpy as np

from utils.cvfpscalc import CvFpsCalc
import saliency_detect
import config


def main():
    print("Starting...\n")
    # カメラ準備 
    cap = cv.VideoCapture(0)
    frame_width = 1920
    frame_height = 1080
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    # GUI準備 
    window_name='Demo'
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)



    # FPS計測モジュール 
    cvFpsCalc = CvFpsCalc()

    while True:
        if config.showfps:
            display_fps = cvFpsCalc.get()

        # カメラキャプチャ 
        ret, frame = cap.read()
        if not ret:
            break
        clipped_frame = frame[0:1040, 640:1280] # カメラキャプチャを縦長にクロップ 2画面なら　[0:1040, 480:1440]

        # 処理 
        out = saliency_detect.saliency_detect(clipped_frame, config.algo)

        # 画面反映 
        if config.showfps:
            cv.putText(out, f"FPS: {int(display_fps)}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow(window_name, out)

        # キー処理 
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()