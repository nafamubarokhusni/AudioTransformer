import os
import tensorflow as tf
import numpy as np
import gc
from AudioTransformer.data import * # type: ignore
from AudioTransformer.evaluation import * # type: ignore
from AudioTransformer.model import * # type: ignore
from AudioTransformer.train import * # type: ignore

class CreateModel():
    def __init__(self, seed=False, multigpu=False):
        super().__init__()
        if seed == True:
            global set_seed
            def set_seed():
                import numpy as np
                import tensorflow as tf
                np.random.seed(212)
                tf.random.set_seed(212)
        if multigpu == True:
            self.strategy = tf.distribute.MirroredStrategy()
        self.model = self.train_data = self.test_data = self.test = None
        self.multigpu = multigpu
        self.D_MODELS = self.SEQ_LENGTH = self.VOCAB_SIZE = self.SPATIAL_SIZE = self.MAX_FRAMES = None
        self.FRAMES_STORAGE_PATH = self.vectorization = None
        self.NUM_CAPTIONS = 40

    def LoadData(self, CAPTIONS_PATH, FRAMES_STORAGE_PATH, train_size, SEQ_LENGTH, 
                 VOCAB_SIZE, SPATIAL_SIZE, MAX_FRAMES, NUM_CAPTIONS, BATCH_SIZE, VIDEOS_PATH):
        captions_mapping, text_data = load_captions_data(CAPTIONS_PATH, SEQ_LENGTH, VIDEOS_PATH)
        train_data, valid_data = train_val_split(captions_mapping, train_size)
        self.vectorization = vectoriz_text(text_data, VOCAB_SIZE, SEQ_LENGTH)
        process_frames(FRAMES_STORAGE_PATH, captions_mapping, SPATIAL_SIZE, MAX_FRAMES)
        train_frame_dirs = [os.path.join(FRAMES_STORAGE_PATH, 
                                         os.path.basename(video).split('.')[0]) for video in captions_mapping.keys()]
        valid_frame_dirs = [os.path.join(FRAMES_STORAGE_PATH, 
                                         os.path.basename(video).split('.')[0]) for video in valid_data.keys()]
        self.train_data = make_dataset_from_frames(train_frame_dirs, list(captions_mapping.values()), 
                                                 self.vectorization, NUM_CAPTIONS, SPATIAL_SIZE, MAX_FRAMES, BATCH_SIZE)
        self.test_data = make_dataset_from_frames(valid_frame_dirs, list(valid_data.values()), 
                                                 self.vectorization, NUM_CAPTIONS, SPATIAL_SIZE, MAX_FRAMES, BATCH_SIZE)
        self.test = valid_data

    def DefineModel(self, ENCODER, DECODER, D_MODELS, NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, 
                    NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L, R_SPATIAL=4, R_TEMPORAL=4, F_SPATIAL=4, F_TEMPORAL=4, TEMPORAL_FIRST=True, ca=False):
        if ca:
            encoder = ENCODER(d_models=D_MODELS, num_heads=NUM_HEADS, 
                                          num_l=NUM_L, max_frames=MAX_FRAMES, spatial_size=SPATIAL_SIZE,R_SPATIAL=R_SPATIAL, 
                                          R_TEMPORAL=R_TEMPORAL, F_SPATIAL=F_SPATIAL, F_TEMPORAL=F_TEMPORAL, TEMPORAL_FIRST=TEMPORAL_FIRST)
        else:
            encoder = ENCODER(d_models=D_MODELS, num_heads=NUM_HEADS, 
                                              num_l=NUM_L, max_frames=MAX_FRAMES, spatial_size=SPATIAL_SIZE)
        encoder.build(input_shape=(None, MAX_FRAMES, SPATIAL_SIZE, SPATIAL_SIZE, 3))
                
        decoder = DECODER(d_models=D_MODELS, num_heads=NUM_HEADS, 
                                vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH, num_l=NUM_L)
        decoder.build([(None, None), (None, NUM_PATCH, D_MODELS), (None, None)])
        model = MainModel(encoder=encoder, decoder=decoder, num_captions_per_video=self.NUM_CAPTIONS)
        return encoder, decoder, model

    def CrossAttention(self, D_MODELS, NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, VOCAB_SIZE, 
                       SEQ_LENGTH, NUM_L, NUM_CAPTIONS=40, R_SPATIAL=4, R_TEMPORAL=4, F_SPATIAL=4, F_TEMPORAL=4, TEMPORAL_FIRST=True):
        del self.model
        NUM_PATCH = int((SPATIAL_SIZE/16)**2 + (MAX_FRAMES//2))
        if self.multigpu == True:
            with self.strategy.scope():
                encoder, decoder, self.model = self.DefineModel(CrossAttention.Encoder, module.Decoder, D_MODELS, NUM_HEADS, MAX_FRAMES, 
                                                                SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L, R_SPATIAL=R_SPATIAL, R_TEMPORAL=R_TEMPORAL, 
                                                                F_SPATIAL=F_SPATIAL, F_TEMPORAL=F_TEMPORAL, TEMPORAL_FIRST=TEMPORAL_FIRST, ca=True)
        else:
            encoder, decoder, self.model = self.DefineModel(CrossAttention.Encoder, module.Decoder, D_MODELS, NUM_HEADS, MAX_FRAMES, 
                                                            SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L, R_SPATIAL=R_SPATIAL, R_TEMPORAL=R_TEMPORAL, 
                                                                F_SPATIAL=F_SPATIAL, F_TEMPORAL=F_TEMPORAL, TEMPORAL_FIRST=TEMPORAL_FIRST, ca=True)  
        self.D_MODELS = D_MODELS
        self.SEQ_LENGTH = SEQ_LENGTH
        self.VOCAB_SIZE = VOCAB_SIZE
        self.SPATIAL_SIZE = SPATIAL_SIZE
        self.MAX_FRAMES = MAX_FRAMES
        self.NUM_CAPTIONS = NUM_CAPTIONS
        trainable_vars = self.model.trainable_variables
        total_params = 0
        for var in trainable_vars:
            var_params = tf.size(var).numpy()
            total_params += var_params
        print(f"Num of trainable parameters: {total_params}")
        # self.model = model

    def fit(self, CAPTIONS_PATH, VIDEOS_PATH, FRAMES_STORAGE_PATH, EPOCHS, BATCH_SIZE, NUM_CAPTIONS=40, test_size=0.2):
        tf.keras.backend.clear_session()
        gc.collect()
        if self.train_data is None or NUM_CAPTIONS != self.NUM_CAPTIONS or FRAMES_STORAGE_PATH != self.FRAMES_STORAGE_PATH:
            self.NUM_CAPTIONS = NUM_CAPTIONS
            self.FRAMES_STORAGE_PATH = FRAMES_STORAGE_PATH
            train_size = 1 - test_size
            self.LoadData(CAPTIONS_PATH, self.FRAMES_STORAGE_PATH, train_size, self.SEQ_LENGTH, 
                         self.VOCAB_SIZE, self.SPATIAL_SIZE, self.MAX_FRAMES, self.NUM_CAPTIONS, BATCH_SIZE, VIDEOS_PATH)
        if self.multigpu == True:
            with self.strategy.scope():
                cross_entropy, early_stopping, optimizer = DefineCompile(self.D_MODELS)
                self.model.compile(optimizer=optimizer, loss=cross_entropy)
        else:
            cross_entropy, early_stopping, optimizer = DefineCompile(self.D_MODELS)
            self.model.compile(optimizer=optimizer, loss=cross_entropy)
                
        history = self.model.fit(self.train_data, 
                                 epochs=EPOCHS, 
                                 validation_data=self.test_data, 
                                 callbacks=[early_stopping])
        return history

    def eval(self):
        evaluation = EvalMetrics(self.model, self.vectorization, self.SEQ_LENGTH, 
                                 self.test_data, self.test, self.FRAMES_STORAGE_PATH, SPATIAL_SIZE=self.SPATIAL_SIZE, max_frames=self.MAX_FRAMES)
        acc, loss = evaluation.acc_loss()
        cider = evaluation.compute_cider()
        return acc, loss, cider