"""Bounded numerical checks; full cached-feature evidence lives in T12 benchmark."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from fast_head import EpochLoop, fit, make_model


class FastNumerics(unittest.TestCase):
    def test_explicit_orders_preserve_weights_slots_and_weighted_losses(self):
        rng = np.random.default_rng(12)
        x, vx = rng.normal(size=(23, 7)).astype("float32"), rng.normal(size=(17, 7)).astype("float32")
        y, vy = rng.integers(9, size=23), rng.integers(9, size=17)
        tf.keras.utils.set_random_seed(12)
        keras, fast = make_model(tf, width=7), make_model(tf, width=7)
        fast.set_weights(keras.get_weights())
        loop = EpochLoop(tf, fast, x, y, vx, vy)
        for _ in range(2):
            order = rng.permutation(len(y)).astype("int32")
            history = keras.fit(x[order], y[order], batch_size=10, shuffle=False,
                                validation_data=(vx, vy), epochs=1, verbose=0).history
            logs = loop.epoch(order)
            for name in ("loss", "val_loss", "accuracy", "val_accuracy"):
                self.assertAlmostEqual(history[name][0], logs[name], delta=1e-6)
            for left, right in zip(keras.get_weights(), fast.get_weights()):
                self.assertLess(float(np.max(np.abs(left - right))), 1e-5)
            for left, right in zip(keras.optimizer.variables, fast.optimizer.variables):
                np.testing.assert_allclose(left.numpy(), right.numpy(), atol=1e-6, rtol=0)

    def test_callbacks_stop_and_retain_distinct_best_and_last_weights(self):
        # Identical rows remove shuffle effects. Opposing val labels force loss
        # to rise, exercising strict best selection and patience=2 at epoch 3.
        x, vx = np.ones((23, 7), np.float32), np.ones((17, 7), np.float32)
        y, vy = np.ones(23, np.int64), np.zeros(17, np.int64)
        outputs = []
        with tempfile.TemporaryDirectory() as temp:
            for trainer in ("keras", "fast"):
                model = make_model(tf, width=7)
                model.set_weights([np.zeros((7, 9), np.float32), np.zeros(9, np.float32)])
                best, csv = Path(temp) / f"{trainer}.weights.h5", Path(temp) / f"{trainer}.csv"
                callbacks = [tf.keras.callbacks.ModelCheckpoint(str(best), monitor="val_loss",
                             save_best_only=True, save_weights_only=True),
                             tf.keras.callbacks.CSVLogger(str(csv)),
                             tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)]
                if trainer == "keras":
                    model.fit(x, y, validation_data=(vx, vy), batch_size=10, epochs=10,
                              callbacks=callbacks, verbose=0)
                else:
                    fit(tf, model, x, y, vx, vy, 10, 10, callbacks)
                last = model.get_weights()
                model.load_weights(best)
                selected = model.get_weights()
                history = pd.read_csv(csv)
                self.assertEqual(len(history), 3)
                self.assertTrue(np.all(np.diff(history.val_loss) > 0))
                self.assertGreater(max(np.max(np.abs(a-b)) for a,b in zip(last, selected)), .001)
                outputs.append((last, selected, history))
            for weights in (0, 1):
                for a, b in zip(outputs[0][weights], outputs[1][weights]):
                    np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
            np.testing.assert_allclose(outputs[0][2], outputs[1][2], atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
