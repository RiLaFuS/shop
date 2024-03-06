import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from ultralytics import YOLO
from io import BytesIO
import numpy as np
import cv2

# タイトルを表示
st.title('いらっしゃいませ')
st.title('グルメマーケットへようこそ')


# 画像ファイルのパス（実際には生成された画像のパスを使用してください）
image_op = Image.open('image2.png')
st.image(image_op, use_column_width=True)

# 説明文を表示
st.write('購入したい商品を選択してください。複数選択も可能です。')

# 商品と個別金額の辞書
prices = {
    'マスカット': 3000, '桃': 400, 'たけのこの里': 230, '芋けんぴ': 300,
    'A5黒毛和牛': 12000, '牡蠣': 800, '寿司': 2500, 'うな重': 1800,
    'クラフトビール': 1500, '日本酒': 2000
}

# 商品一覧の選択ボタンを表示
selected_items = st.multiselect("商品一覧", list(prices.keys()))

# 選択された商品の個別金額と合計金額を計算して表示
total_price = sum(prices[item] for item in selected_items)
for item in selected_items:
    st.write(f"{item}: {prices[item]}円")
st.write(f"合計金額: {total_price}円")


# サイドバーにダウンロードボタンを設置
st.sidebar.title("支払い方法の選択")
st.sidebar.write("いずれかの画像をダウンロードしてください。")
# ダウンロードする動画ファイルのリスト
image_files = ["coin.jpg", "paper.jpg"]

# 各動画ファイルに対してダウンロードボタンを追加
for image_file_name in image_files:
    image_file_path = Path(image_file_name)

    # ファイルが存在するかどうかを確認
    if image_file_path.is_file():
        # ファイルを読み込む
        with open(image_file_path, "rb") as file:
            # サイドバーにダウンロードボタンを追加
            st.sidebar.download_button(
                label=f"{image_file_name}をダウンロード",
                data=file,
                file_name=image_file_name,
                mime="image/jpg",
            )
    else:
        st.sidebar.write(f"{image_file_name}が見つかりません。")

st.title('支払い')
uploaded_img = st.file_uploader("ダウンロードした画像をアップロードしてください。", type=['jpg', 'png'])

model = YOLO('best.pt')

if  uploaded_img is not None:
    # プログレスバーを表示
    progress_bar = st.progress(0)

    # ファイルを読み込み
    bytes_data  = uploaded_img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 物体検知の実行
    results = model(cv2_img, conf=0.3, iou=0.95)

    # プログレスバーを100%に更新
    progress_bar.progress(100)

    # 検出結果を描画
    output_img = results[0].plot(labels=True, conf=True)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # 画像を表示
    st.image(output_img)

    # アップロードされた画像の名前を取得
    image_name = uploaded_img.name

    # 画像の名前に基づいて金額を表示
    if image_name == "coin.jpg":
        st.write("774円頂戴いたします。・・・")
    elif image_name == "paper.jpg":
        st.write("5234円頂戴いたします。・・・")

    # 画像の名前に基づいて条件分岐
    if image_name == "coin.jpg":
        difference = 774 - total_price
    elif image_name == "paper.jpg":
        difference = 5234 - total_price
    else:
        st.write("アップロードされた画像が支払いに対応していません。")
        difference = None

    # 差額に基づいてメッセージを表示
    if difference is not None:
        if difference < 0:
            st.write("残念ですが、お金が足りません。またのご来店をお待ちしております。")
        else:
            st.write(f"ありがとうございます。おつりは{difference}円です。またのご来店をお待ちしております。")

    st.markdown("""
        <style>
        .big-font {
            font-size:17px !important;
            font-weight: bold;
        }
        </style>
        <div class='big-font'>
            <br>
            <br>
            参考情報（2024.3.1時点）：<br>
            <br>
            硬貨<br>
            1 CNY（中国元） : 21円<br>
            1 HKD（香港ドル） : 19円<br>
            1 cent（セント） : 2円<br>
            1 kr（ノルウェークローネ） : 14円<br>
            10 cent（セント） : 15円<br>
            2 EUR（ユーロ） : 325円<br>
            20 kr（ノルウェークローネ） : 285円<br>
            5 jiao（中国角） : 9円<br>
            50 cent（セント） : 75円<br>
            50 lp（クロアチアルピー） : 9円<br>
            <br>
            紙幣<br>
            10 USD（USドル） : 1505円<br>
            20 EUR（ユーロ） : 3249円<br>
            100 NTD（台湾ドル） : 475円<br>
            500 rupee（スリランカルピー） : 5円
        </div>
        """, unsafe_allow_html=True)