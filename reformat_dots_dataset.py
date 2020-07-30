import numpy as np
import os


class DotsDataset():
    def __init__(self, db_path=('/data/dots/',), noisy=True):
        self.data_dims = [64, 64, 3]
        self.name = "dots"
        self.noisy = noisy
        self.batch_size = 100
        self.db_path = db_path

        self.db_files = [os.listdir(path) for path in db_path]
        assert np.min([len(files) for files in self.db_files]) == np.max([len(files) for files in self.db_files])
        self.train_db_files = self.db_files
        # print(self.db_files)

        self.train_data_ptr = 0
        self.train_batch_ptr = -1
        self.batch_cache = {}
        self.train_batch = self.load_new_data()

        self.train_size = len(self.db_files) * 8192
        self.range = [0.0, 1.0]

    def load_new_data(self):
        self.train_batch_ptr += 1
        if self.train_batch_ptr == len(self.train_db_files[0]):
            self.train_batch_ptr = 0

        if self.train_batch_ptr not in self.batch_cache:
            images_list = []
            for dtype in range(len(self.train_db_files)):
                filename = os.path.join(self.db_path[dtype], self.train_db_files[dtype][self.train_batch_ptr])
                img = np.load(filename)['images']
                if self.noisy:
                    img += np.random.normal(loc=0, scale=0.03, size=img.shape)
                    img = 1.0 - np.abs(1.0 - img)
                    img = np.abs(img)
                images_list.append(img)
            images = np.concatenate(images_list, axis=0)
            self.batch_cache[self.train_batch_ptr] = {'images': images}

        return self.batch_cache[self.train_batch_ptr]['images']

    def move_train_ptr(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_data_ptr = self.train_data_ptr
        self.train_data_ptr += batch_size
        if self.train_data_ptr > self.train_batch.shape[0]:
            self.train_data_ptr = batch_size
            prev_data_ptr = 0
            self.train_batch = self.load_new_data()
        return prev_data_ptr

    def next_batch(self, batch_size=None):
        prev_data_ptr = self.move_train_ptr(batch_size)
        return self.train_batch[prev_data_ptr:self.train_data_ptr, :, :, :]

    def reset(self):
        self.train_data_ptr = 0
        self.train_batch_ptr = -1
        self.train_batch = self.load_new_data()


if __name__ == '__main__':
    import tensorflow as tf
    dataset = DotsDataset(db_path=['/home/helvellyn/dotsdata/dots/6_dots'], noisy=False)
    i = 0
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    outFilePath = "6_dots.tfrecords"
    writer = tf.python_io.TFRecordWriter(outFilePath, options=options)
    i = 0
    for image in dataset.train_batch:
        i += 1
        if i % 100 == 0:
            print(i)
        raw = (image*255).astype(np.uint8).reshape(-1)
        rawb = [b.tobytes() for b in raw]
        record_bytes = tf.train.Example(features=tf.train.Features(feature={"image":tf.train.Feature(bytes_list=tf.train.BytesList(value=rawb))}))
        writer.write(record_bytes.SerializeToString())
    print("done!")
    print(i)
