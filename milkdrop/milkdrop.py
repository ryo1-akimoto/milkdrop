# ライブラリのインポート
import cv2
import numpy as np

# 画像の読み込み
img = cv2.imread('images/milkdrop.bmp')

# 画像の表示
cv2.imshow('milkdorp', img)
cv2.waitKey(0)

# 画像のグレースケール化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 画像の2値化
ret, thresh = cv2.threshold(img_gray, 136, 255, cv2.THRESH_BINARY)

# ノイズ除去
thresh = cv2.medianBlur(thresh, 5)

# 輪郭の抽出
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭の描画
cv2.drawContours(img, contours, -1, color=(255, 255, 255), thickness=2)

# 面積が最大の輪郭を取得
contour = max(contours, key=lambda x: cv2.contourArea(x))

# マスク画像の作成
mask = np.zeros_like(thresh)
cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)

# 画像の合成
img_and = cv2.bitwise_and(img, img, mask=mask)

# 画像の表示
cv2.imshow('masked_milkdrop', img_and)
cv2.waitKey(0)