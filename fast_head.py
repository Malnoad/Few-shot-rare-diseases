"""Device-resident epoch loop using Keras's own loss, metrics and Adam steps.

No optimizer algebra is reimplemented. An explicit order is accepted for the
equivalence gate; production uses a fresh TensorFlow permutation every epoch.
"""


def make_model(tf, width=2048, classes=9, lr=.001):
    model = tf.keras.Sequential([tf.keras.Input((width,)),
                                 tf.keras.layers.Dense(classes, activation="softmax")])
    model.compile(tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class EpochLoop:
    def __init__(self, tf, model, x, y, val_x, val_y, batch=10, jit=False):
        self.tf, self.model = tf, model
        self.x, self.y = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
        self.vx, self.vy = tf.convert_to_tensor(val_x), tf.convert_to_tensor(val_y)
        self.batch = batch
        # Build metric/optimizer variables outside the graph's loop body.
        model.optimizer.build(model.trainable_variables)
        model.test_on_batch(val_x[:batch], val_y[:batch])
        model.reset_metrics()
        self.train = tf.function(self._train, jit_compile=jit)
        self.validate = tf.function(self._validate, jit_compile=jit)

    def _train(self, order):
        tf = self.tf
        for start in tf.range(0, tf.shape(order)[0], self.batch):
            indices = order[start:start + self.batch]
            self.model.train_step((tf.gather(self.x, indices), tf.gather(self.y, indices)))
        return self.model.get_metrics_result()

    def _validate(self):
        tf = self.tf
        for start in tf.range(0, tf.shape(self.vy)[0], self.batch):
            self.model.test_step((self.vx[start:start + self.batch], self.vy[start:start + self.batch]))
        return self.model.get_metrics_result()

    def epoch(self, order=None):
        self.model.reset_metrics()
        if order is None:
            order = self.tf.random.shuffle(self.tf.range(len(self.y)))
        train = {k: float(v) for k, v in self.train(order).items()}
        self.model.reset_metrics()
        val = {"val_" + k: float(v) for k, v in self.validate().items()}
        return dict(train, **val)


def fit(tf, model, x, y, val_x, val_y, epochs, batch, callbacks):
    loop = EpochLoop(tf, model, x, y, val_x, val_y, batch)
    cbs = tf.keras.callbacks.CallbackList(callbacks, model=model, epochs=epochs,
                                         steps=(len(y) + batch - 1) // batch)
    model.stop_training = False
    cbs.on_train_begin()
    try:
        for epoch in range(epochs):
            cbs.on_epoch_begin(epoch)
            logs = loop.epoch()
            cbs.on_epoch_end(epoch, logs)
            print(f"Epoch {epoch + 1}/{epochs}: {logs}", flush=True)
            if model.stop_training:
                break
    finally:
        cbs.on_train_end()
