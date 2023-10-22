# CycleGAN

このプロジェクトは、オリジナルの画像を油絵風に変換するためのものです。

## 使用方法

1. **フォルダを追加**:

   以下のフォルダを「cyclegan」内に追加してください：

   - `datasets`
   - `images`
   - `original_image`
   - `result`

2. **フォルダに画像を保存**:

   画像を1枚「original_image」フォルダに保存します。

3. **cyclegan_test.pyを実行**:

   「cyclegan_test.py」スクリプトを実行します。実行後、変換された画像は「result」フォルダに保存されます。

## 補足

1. **Spring Bootとの連携**:

   Spring Boot（[https://github.com/soda143/springboot](https://github.com/soda143/springboot)）と連携することで、画像の変換を簡単に行うことができます。

2. **ソースコードの説明**:

   - 「cyclegan.py」: CycleGANの生成モデルが記述されたファイルです。「datasets」 フォルダ内に「oil painting」フォルダを作成し、その中に「trainA」「trainB」「testA」「testB」のフォルダを作成することで、画像の学習を行えます。

   - 「data_loader.py」: 「cyclegan.py」 でインポートするファイルです。

   - 「cyclegan_test.py」: 「weights」フォルダの重みを使用してオリジナルの画像を学習するためのファイルです。
