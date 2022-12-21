import tensorflow as tf 
from tensorflow import keras
from student_mod_tf.model import MobileViT_XS

def get_encoder():
    
    mobile = MobileViT_XS(input_shape=(256,256,3))
    last_layer = mobile.layers[-2].output

    model = keras.Model(mobile.input, last_layer)
    return model


def get_projection_head(dims=[384, 1024*2, 1024*7, 1024*7]):
    return keras.Sequential(
        [
            keras.Input(shape=(dims[0],)),
            keras.layers.Dense(dims[1]),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(dims[2]),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(dims[3]),
        ],
        name="projection_head",
    )
    
def get_linear_probe(z_dim):
    return keras.Sequential(
        [
            keras.layers.Input(shape=(z_dim,)), 
            keras.layers.Dense(7)],
        name="linear_probe"
    )

def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n-1, n+1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])

def normalize_repr(z):
    z_norm = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    return z_norm

def compute_loss(z_a, z_b, lambd=5e-3):
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    # Normalize the representations along the batch dimension.
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    # Cross-correlation matrix.
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size

    # Loss.
    on_diag = tf.linalg.diag_part(c) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2))
    off_diag = off_diagonal(c)
    off_diag = tf.reduce_sum(tf.pow(off_diag, 2))
    loss = on_diag + (lambd * off_diag)
    return loss    

class BarlowModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = get_encoder()
        z_dim = 384
        self.projection_head = get_projection_head([z_dim, 1024*3, 1024*8, 1024*8])
        self.linear_probe = get_linear_probe(z_dim)


    def compile(self, main_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.main_optimizer = main_optimizer
        self.probe_optimizer = probe_optimizer

        self.main_loss_tracker = keras.metrics.Mean(name="c_loss")
        # self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.main_loss_tracker,
            # self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def train_step(self, batch):
        y_a, y_b = batch
        
        with tf.GradientTape() as tape:
            z_a_1, z_b_1 = self.encoder(y_a, training=True), self.encoder(y_b, training=True)
            z_a_2, z_b_2 = self.projection_head(z_a_1, training=True), self.projection_head(z_b_1, training=True)
            
            main_loss = compute_loss(z_a_2, z_b_2)

        gradients = tape.gradient(main_loss, 
                self.encoder.trainable_weights + self.projection_head.trainable_weights)
        
        self.main_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )

        self.main_loss_tracker.update_state(main_loss)
        
        return {"loss": self.main_loss_tracker.result()}
    
    def test_step(self, batch):
        imgs, labels = batch
        with tf.GradientTape() as tape:
            features = self.encoder(imgs, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_accuracy.update_state(labels, class_logits)
        
        return {"acc": self.probe_accuracy.result()} 
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        print("CALL")

