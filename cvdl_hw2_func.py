import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def background_subtraction():
    # 選擇影片文件
    video_path = "./Dataset_CvDl_Hw2/Q1/traffic.mp4"
    
    cap = cv2.VideoCapture(video_path)

    # 創建背景减除器
    history = 1000
    dist2Threshold = 650
    subtractor = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 模糊處理
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # 應用背景减除器獲得背景遮罩
        mask = subtractor.apply(blurred_frame)

        # 通過按位與操作得到僅包含移動對象的畫面
        result_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # 顯示原始影像
        cv2.imshow("Original", frame)

        # 顯示背景遮罩
        cv2.imshow("Background Mask", mask)

        # 顯示處理後的結果
        cv2.imshow("Result", result_frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def optical_flow_preprocessing():
    video_path = "./Dataset_CvDl_Hw2/Q2/optical_flow.mp4"
    
    cap = cv2.VideoCapture(video_path)

    # 讀取第一帧
    ret, frame = cap.read()

    if not ret:
        print("Unable to read the video.")
        return

    # 參數調整
    max_corners = 1
    quality_level = 0.3
    min_distance = 7
    block_size = 7

    # 將第一帧轉為灰度
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用goodFeaturesToTrack檢測特徵點
    corners = cv2.goodFeaturesToTrack(gray_frame, max_corners, quality_level, min_distance, blockSize=block_size)

    if corners is not None:
        # 將特徵點轉換為整數
        corners = np.int0(corners)

        # 在原始圖像上畫出紅色的十字標記
        for corner in corners:
            x, y = corner.ravel()
            cv2.line(frame, (x - 20, y), (x + 20, y), (0, 0, 255), 2)  # 水平線
            cv2.line(frame, (x, y - 20), (x, y + 20), (0, 0, 255), 2)  # 垂直線

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # 顯示結果
        cv2.imshow("Nose Point Detection", small_frame)
        cv2.waitKey(0)
    else:
        print("No corners detected in the first frame.")

    cap.release()
    cv2.destroyAllWindows()
    
    return x, y
    
def optical_flow_video_tracking():
    cap = cv2.VideoCapture('./Dataset_CvDl_Hw2/Q2/optical_flow.mp4')

    # ShiTomasi角點檢測的參數
    feature_params = dict(maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Lucas-Kanade光流的參數
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 產生一些隨機顏色
    color = np.random.randint(0, 255, (1, 3))

    # 讀取第一幀並在其中找到特徵點
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # 創建用於繪製的遮罩圖像
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()

        if not ret:
            print('No frames grabbed!')
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 計算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 選擇好的點
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # 畫出軌跡
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff

        if k == 27:
            break

        # 更新先前的幀和特徵點
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    
def pca():
    # 讀取ARGB圖像
    image = cv2.imread("./Dataset_CvDl_Hw2/Q3/logo.jpg", cv2.IMREAD_UNCHANGED)

    # 1. 轉換為灰度圖
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 正規化灰度圖像
    normalized_image = gray_image / 255.0

    # 獲取圖像尺寸
    height, width = normalized_image.shape

    # 初始化PCA
    pca = PCA()

    # 初始化最小誤差和最小主成分數量
    min_error = float('inf')
    min_components = 0

    # 迭代主成分數量，計算重建誤差
    for n in range(1, min(height, width) + 1):
        pca.n_components = n
        transformed_image = pca.fit_transform(normalized_image)
        reconstructed_image = pca.inverse_transform(transformed_image)
        mse = mean_squared_error(normalized_image, reconstructed_image)
        
        # 判斷是否滿足條件
        if mse <= 3.0 and mse < min_error:
            min_error = mse
            min_components = n

    # 顯示最小主成分數量
    print("Minimum components for MSE <= 3.0:", min_components)

    # 重新設置PCA
    pca.n_components = min_components
    transformed_image = pca.fit_transform(normalized_image)
    reconstructed_image = pca.inverse_transform(transformed_image)

    # 顯示原始圖像和重建圖像
    plt.subplot(1, 2, 1)
    plt.imshow(normalized_image, cmap='gray')
    plt.title("Gray Scaled Image")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"Reconstructed Image with {min_components} Components")

    plt.show()


# def main():
#     while True:
#         select = int(input('Choose question number: '))
#         if select == 1:        
#             background_subtraction()
#         elif select == 2:
#             optical_flow_preprocessing()
            # optical_flow_video_tracking()
#         elif select == 3:
#             pca()
#         elif select == 0:
#             break

if __name__ == "__main__":
    # main()
    pca()