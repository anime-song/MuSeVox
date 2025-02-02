import tensorflow as tf


def glu(x, axis=-1):
    """
    GLU (Gated Linear Unit) の実装。
    入力テンソル x を指定軸で2分割し、前半と後半に分けて
    前半 * sigmoid(後半) を返す。
    """
    a, b = tf.split(x, num_or_size_splits=2, axis=axis)
    return a * tf.sigmoid(b)


class ConvolutionModule(tf.keras.layers.Layer):
    """
    SD block 内で利用する Convolution Module の実装

    Args:
        channels (int): 入出力のチャネル数。
        depth (int): Residual branch 内での層の数。
        compress (float): チャネル圧縮率。hidden_size = channels / compress として計算。
        kernel (int): 畳み込みのカーネルサイズ（奇数である必要があります）。
    """

    def __init__(self, channels, depth=2, compress=4, kernel=3, **kwargs):
        super(ConvolutionModule, self).__init__(**kwargs)
        assert kernel % 2 == 1, "Kernel size must be odd."
        self.depth = abs(depth)
        hidden_size = channels // compress

        # 各層ごとに residual ブロックを構築
        self.layers_list = []
        for _ in range(self.depth):
            # ここでは PyTorch 版の nn.GroupNorm(1, channels) に近い動作として
            # tf.keras.layers.LayerNormalization を利用（channels は最後の次元）。
            norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)
            # Conv1D: 入力 channels -> hidden_size * 2
            conv1 = tf.keras.layers.Conv1D(filters=hidden_size * 2, kernel_size=kernel, padding="same")
            # GLU を適用（出力チャネルは hidden_size に半減）
            # 次に、グループ化畳み込み（depthwise convolution に相当）を実施します。
            # tf.keras.layers.Conv1D は TensorFlow 2.10 以降で groups 引数が利用可能です。
            conv2 = tf.keras.layers.Conv1D(filters=hidden_size, kernel_size=kernel, padding="same", groups=hidden_size)
            norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)
            swish = tf.keras.layers.Activation("swish")
            # 最後、1x1 畳み込みでチャネル数を元に戻す
            conv3 = tf.keras.layers.Conv1D(filters=channels, kernel_size=1, padding="same")

            # 1 つの residual ブロックとして Sequential でまとめる
            block = tf.keras.Sequential(
                [
                    norm1,
                    conv1,
                    tf.keras.layers.Lambda(lambda x: glu(x, axis=-1)),
                    conv2,
                    norm2,
                    swish,
                    conv3,
                ]
            )
            self.layers_list.append(block)

    def call(self, x):
        # x の shape: (batch, length, channels)
        for block in self.layers_list:
            # 残差接続: 入力にブロックの出力を加算
            x = x + block(x)
        return x


class FusionLayer(tf.keras.layers.Layer):
    """
    FusionLayer (decoder 内の融合層)

    Args:
      channels (int): 入力チャンネル数（Conv の前は C だが、内部では x.repeat(...) により C*2 となる）
      kernel_size (int, optional): 畳み込みのカーネルサイズ。デフォルトは 3。
      stride (int, optional): 畳み込みのストライド。デフォルトは 1。
      padding (int, optional): 畳み込みのパディング。デフォルトは 1。
    """

    def __init__(self, channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # PyTorch の nn.Conv2d(channels*2, channels*2, kernel_size, stride, padding)
        # を再現するため、padding > 0 の場合は事前に ZeroPadding2D を適用し、
        # conv2d は 'valid' パディングで実施する
        if self.padding > 0:
            self.pad = tf.keras.layers.ZeroPadding2D(padding=(padding, padding), data_format="channels_last")
        else:
            # パディングが不要な場合は恒等変換
            self.pad = lambda x: x

        self.conv = tf.keras.layers.Conv2D(
            filters=channels * 2,  # 出力チャネル数は channels*2
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="valid",  # 事前に pad しているので 'valid'
            data_format="channels_last",
        )

    def call(self, x, skip=None):
        """
        Args:
          x: 入力テンソル。shape = (B, T, Fr, C)
          skip: スキップ接続用のテンソル（x と同shape）。None でなければ x に加算する。
        Returns:
          Tensor: GLU を適用した出力 (shape = (B, T, Fr, C))
        """
        # skip が与えられていれば、element-wise に加算
        if skip is not None:
            x = x + skip

        # チャンネル数を2倍に複製 (channels_last なので最後の次元を2倍)
        # 例: shape (B, T, Fr, C) → (B, T, Fr, C*2)
        x = tf.tile(x, multiples=[1, 1, 1, 2])

        # パディング適用（padding > 0 の場合）
        x = self.pad(x)

        # 畳み込み (Conv2D は入力も出力も shape (B, T, Fr, channels*2) となる)
        x = self.conv(x)

        # GLU: 出力チャンネルを半分に分割し、a * sigmoid(b) を計算
        x = glu(x, axis=-1)
        return x


class SDLayer(tf.keras.layers.Layer):
    """
    Sparse Down-sample Layer（SDLayer）の TensorFlow 実装です。

    Args:
      channels_out (int): 出力チャンネル数
      band_configs (dict): 周波数帯ごとの設定を含む辞書。
                           キーは 'low', 'mid', 'high' で、それぞれの値は
                           辞書 { 'SR': サンプル比率, 'stride': ストライド, 'kernel': カーネルサイズ } を持つ。
    """

    def __init__(self, channels_out, band_configs, **kwargs):
        super(SDLayer, self).__init__(**kwargs)

        self.convs = []  # 各帯域用の畳み込み層リスト
        self.strides = []  # 各帯域のストライド値
        self.kernels = []  # 各帯域のカーネルサイズ

        # 各帯域ごとに Conv2D レイヤーを生成
        # ※ 入力が (B, T, Fr, C) で、Fr 軸に沿って処理を行うので、
        #     カーネルサイズは (1, kernel)、ストライドは (1, stride) とする。
        for key in band_configs:
            config = band_configs[key]
            self.convs.append(
                tf.keras.layers.Conv2D(
                    filters=channels_out,
                    kernel_size=(1, config["kernel"]),
                    strides=(1, config["stride"]),
                    padding="valid",  # パディングは後で手動で実施
                    data_format="channels_last",
                )
            )
            self.strides.append(config["stride"])
            self.kernels.append(config["kernel"])

        # 分割比率の保存（低域と中域の比率）
        self.SR_low = band_configs["low"]["SR"]
        self.SR_mid = band_configs["mid"]["SR"]

    def call(self, x):
        """
        入力テンソル x の周波数軸（axis=2）を分割し、各帯域に対して対応する畳み込みを適用します。

        Args:
          x (Tensor): 入力テンソル。形状は (B, T, Fr, C)

        Returns:
          outputs (list): 各帯域ごとの畳み込み結果のリスト
          original_lengths (list): 各帯域の元の周波数軸の長さのリスト
        """
        # 入力の shape (B, T, Fr, C) から各次元を取得
        Fr = tf.shape(x)[2]  # 周波数軸

        # 周波数軸での分割インデックスを計算
        Fr_float = tf.cast(Fr, tf.float32)
        end_low = tf.cast(tf.math.ceil(Fr_float * self.SR_low), tf.int32)
        end_mid = tf.cast(tf.math.ceil(Fr_float * (self.SR_low + self.SR_mid)), tf.int32)
        splits = [(0, end_low), (end_low, end_mid), (end_mid, Fr)]

        outputs = []
        original_lengths = []

        # 各帯域ごとに処理を実施
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            # 周波数軸（axis=2）の部分を抽出
            # 入力 shape: (B, T, Fr, C) → 抽出後: (B, T, band_length, C)
            extracted = x[:, :, start:end, :]
            band_length = end - start
            original_lengths.append(band_length)

            current_length = tf.shape(extracted)[2]

            # ストライドに合わせたパディング量の計算
            # stride が 1 の場合は kernel - 1 のパディング量
            if stride == 1:
                total_padding = kernel - stride
            else:
                mod = tf.math.mod(current_length, stride)
                total_padding = tf.math.mod(stride - mod, stride)

            # 周波数軸（axis=2）に対する左右のパディング量を決定（均等にパディング）
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            # channels_last の場合、パディング設定は以下の通り
            # バッチ, 時間, 周波数, チャンネル の各軸に対して [前, 後] のパディング量を指定
            paddings = [
                [0, 0],  # バッチ次元
                [0, 0],  # 時間次元
                [pad_left, pad_right],  # 周波数次元
                [0, 0],  # チャンネル次元
            ]
            padded = tf.pad(extracted, paddings, mode="constant")

            # 畳み込み層の適用
            out = conv(padded)
            outputs.append(out)

        return outputs, original_lengths


class SULayer(tf.keras.layers.Layer):
    """
    Sparse Up-sample Layer（SULayer）の TensorFlow 実装です。

    Args:
      channels_out (int): 出力チャンネル数
      band_configs (dict): 周波数帯ごとの設定を含む辞書。
                           キーは 'low', 'mid', 'high' で、それぞれの値は
                           辞書 { 'SR': サンプル比率, 'stride': ストライド, 'kernel': カーネルサイズ } を持つ。
    """

    def __init__(self, channels_out, band_configs, **kwargs):
        super(SULayer, self).__init__(**kwargs)

        # 各バンドごとに Conv2DTranspose レイヤーを生成
        # ※ 入力テンソルは (B, T, Fr, C) で、周波数軸に沿ったアップサンプリングを行うため、
        #     カーネルサイズは (1, kernel)、ストライドは (1, stride) とする。
        self.convtrs = []
        for _, config in band_configs.items():
            self.convtrs.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=channels_out,
                    kernel_size=(1, config["kernel"]),
                    strides=(1, config["stride"]),
                    padding="valid",
                    data_format="channels_last",
                )
            )

    def call(self, x, lengths, original_lengths):
        """
        Args:
          x: 入力テンソル。shape = (B, T, Fr, C)
          lengths: 各バンドに対応する現在の周波数軸の長さリスト（例: [low_length, mid_length, high_length]）
          original_lengths: 各バンドの元の周波数軸の長さリスト

        Returns:
          Tensor: 各バンドごとの Conv2DTranspose 出力をトリミングして連結した結果（周波数軸方向に連結）
        """
        # 周波数軸（axis=2）のスライス開始・終了位置を決定
        splits = [(0, lengths[0]), (lengths[0], lengths[0] + lengths[1]), (lengths[0] + lengths[1], None)]

        outputs = []
        # 各バンドごとに処理
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            # 周波数軸（axis=2）に沿ってスライス
            if end is None:
                x_slice = x[:, :, start:, :]  # shape: (B, T, band_length, C)
            else:
                x_slice = x[:, :, start:end, :]
            # 転置畳み込みを適用（出力 shape: (B, T, new_Fr, channels_out)）
            out = convtr(x_slice)

            # 出力の周波数長を取得し、元の長さに合わせて左右対称にトリミング
            current_Fr_length = tf.shape(out)[2]
            origin_length = tf.cast(original_lengths[idx], tf.int32)
            # 出力と元の長さとの差を算出し、両側から切り落とすための開始位置を計算
            dist = tf.abs(origin_length - current_Fr_length) // 2
            # トリミング（axis=2: 周波数軸）
            trimmed_out = out[:, :, dist : dist + origin_length, :]
            outputs.append(trimmed_out)

        # 各バンドの出力を周波数軸（axis=2）に沿って連結
        x_concat = tf.concat(outputs, axis=2)
        return x_concat


class SDBlock(tf.keras.layers.Layer):
    """
    SDBlock (Sparse Down-sample block) の実装例

    Args:
      channels_out (int): 出力チャネル数。
      band_configs (dict): SDlayer 用のバンド分割などの設定。
      conv_config (dict): 各バンドに適用する ConvolutionModule の設定（辞書として渡す）。
      depths (list of int): 各バンドごとに適用する ConvolutionModule の深さのリスト（例: [3, 2, 1]）。
      kernel_size (int): global convolution のカーネルサイズ（奇数であること）。
    """

    def __init__(self, channels_out, band_configs={}, conv_config={}, depths=[3, 2, 1], kernel_size=3, **kwargs):
        super(SDBlock, self).__init__(**kwargs)
        # SDlayer の生成 (入力 shape: (B, T, Fr, C))
        self.sd_layer = SDLayer(channels_out, band_configs)
        # 各バンド用の ConvolutionModule を depths に合わせて生成
        self.conv_modules = [ConvolutionModule(channels_out, depth, **conv_config) for depth in depths]
        # global convolution (padding を 'same' にすることで、奇数カーネルの場合は出力サイズを維持)
        self.globalconv = tf.keras.layers.Conv2D(
            filters=channels_out,
            kernel_size=kernel_size,
            strides=1,
            padding="same",  # PyTorch では padding=(kernel_size-1)//2 としていた
            data_format="channels_last",
        )

    def call(self, x):
        # x の shape: (B, T, Fr, C)
        bands, original_lengths = self.sd_layer(x)  # 各バンドの shape は (B, T, f, C)
        processed_bands = []
        for conv, band in zip(self.conv_modules, bands):
            # --- 各バンドに対する処理 ---
            # 現在の band shape: (B, T, f, C)
            # 1. 転置して (B, f, T, C)
            band_perm = tf.transpose(band, perm=[0, 2, 1, 3])
            # 2. reshape して、バッチ次元と f を合成 → (B*f, T, C)
            B = tf.shape(band_perm)[0]
            f = tf.shape(band_perm)[1]
            T = tf.shape(band_perm)[2]
            C = tf.shape(band_perm)[3]
            band_reshaped = tf.reshape(band_perm, shape=[-1, T, C])
            # 3. ConvolutionModule を適用
            conv_out = conv(band_reshaped)  # 出力 shape: (B*f, T, C)
            # 4. reshape して元に戻す → (B, f, T, C)
            out_reshaped = tf.reshape(conv_out, shape=[B, f, T, C])
            # 5. 転置して (B, T, f, C)
            out_perm = tf.transpose(out_reshaped, perm=[0, 2, 1, 3])
            # 6. GELU 活性化の適用
            processed = tf.nn.gelu(out_perm)
            processed_bands.append(processed)

        # lengths: 各バンドの周波数方向のサイズ (axis=2)
        lengths = [tf.shape(band)[2] for band in processed_bands]
        # 全バンドを周波数軸 (axis=2) で連結 → full_band の shape: (B, T, f_total, C)
        full_band = tf.concat(processed_bands, axis=2)
        skip = full_band
        # global convolution の適用 (入力 shape: (B, T, f_total, C))
        output = self.globalconv(full_band)

        return output, skip, lengths, original_lengths


if __name__ == "__main__":
    dummy_input = tf.random.normal((4, 50, 100, 64))
    inputs = tf.pad(dummy_input, [[0, 0], [0, 3], [0, 0], [0, 0]], mode="REFLECT")
    split_inputs = tf.split(dummy_input, num_or_size_splits=2 * 2, axis=-1)
    spec = tf.math.exp(tf.stack(split_inputs[:2], axis=-1))
    spec = tf.clip_by_value(spec, spec, 1e3)  # nan回避
    phase = tf.stack(split_inputs[2:], axis=-1)
    print(tf.shape(spec), tf.shape(phase))
