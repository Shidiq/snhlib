import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import BatchNormalization, Dense, Input, LeakyReLU
from tensorflow.keras.models import Model


class FeatureAutoEncoder:
    """_summary_"""

    def __init__(
        self,
        n_target: int,
        test_size=0.2,
        epochs=200,
        batch_size=16,
        random_state=None,
        verbose=2,
    ) -> None:
        self.n_bottleneck = n_target
        self.test_size = test_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.history_ = None
        self.encoder_ = None
        return

    def fit(self, X_train, X_test) -> None:

        t = MinMaxScaler()
        t.fit(X_train)
        X_train = t.transform(X_train)
        X_test = t.transform(X_test)

        # define encoder
        n_inputs = X_train.shape[1]
        visible = Input(shape=(n_inputs,))

        # encoder level 1
        e = Dense(n_inputs * 2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)

        # encoder level 2
        e = Dense(n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)

        # bottleneck
        n_bottleneck = self.n_bottleneck
        bottleneck = Dense(n_bottleneck)(e)

        # define decoder level 1
        d = Dense(n_inputs)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)

        # output layer
        output = Dense(n_inputs, activation="linear")(d)

        # define autoencoder model
        model = Model(inputs=visible, outputs=output)

        # compile autoencoder model
        model.compile(optimizer="adam", loss="mse")

        self.history_ = model.fit(
            X_train,
            X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=(X_test, X_test),
        )

        self.encoder_ = Model(inputs=visible, outputs=bottleneck)
        return

    @property
    def plot_history_(self):
        plt.plot(self.history_.history["loss"], label="train")
        plt.plot(self.history_.history["val_loss"], label="test")
        plt.legend()
        plt.show()
