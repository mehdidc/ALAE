# Adapted from :
# - https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py
# - https://docs.nvidia.com/deeplearning/dali/archives/dali_012_beta/dali-master-branch-user-guide/docs/examples/dataloading_tfrecord.html

import torch
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec


class TFRecordPipeline(Pipeline):
    def __init__(
        self,
        tfrecord_files,
        idx_files,
        batch_size,
        num_threads,
        device_id,
        image_size=224,
        dali_device="cpu",
        rank=0,
        world_size=1,
        data_augmentation=True,
        shuffle=True,
        shuffle_buffer_size=8192,
        prefetch_queue_depth=1,
        shard=False,
    ):
        super().__init__(batch_size, num_threads, device_id)
        features = {
            "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
            "image/class/text": tfrec.FixedLenFeature([], tfrec.string, ""),
        }
        if shard:
            sharding ={
                "shard_id":rank,
                "num_shards":world_size
            }
        else:
            sharding = {}
        self.input = ops.TFRecordReader(
            #more info about the params here:
            #https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.ops.TFRecordReader
            path=tfrecord_files, 
            index_path=idx_files, 
            features=features,
            initial_fill=shuffle_buffer_size,
            random_shuffle=shuffle,
            # prefetch_queue_depth=prefetch_queue_depth,
            # read_ahead=True,
            **sharding,
        )
        self.data_augmentation = data_augmentation
        if dali_device == "cpu":
            self.decode = ops.ImageDecoder(device=dali_device, output_type=types.RGB)
        else:
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoder(
                device="mixed",
                output_type=types.RGB,
                device_memory_padding=211025920,
                host_memory_padding=140544512,
            )

        if data_augmentation:
            self.res = ops.RandomResizedCrop(
                device=dali_device,
                size=(image_size, image_size),
                interp_type=types.INTERP_LINEAR,
                random_aspect_ratio=[0.75, 4.0 / 3.0],
                random_area=[0.08, 1.0],
                num_attempts=100,
            )

            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(image_size, image_size),
                image_type=types.RGB,
            )
            self.coin = ops.CoinFlip(probability=0.5)
        else:
            self.res = ops.Resize(device=dali_device, resize_shorter=size)
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(image_size, image_size),
                image_type=types.RGB,
            )

    def define_graph(self):
        inputs = self.input()
        images = self.decode(inputs["image/encoded"])
        images = self.res(images)
        mirror = self.coin() if self.data_augmentation else None
        output = self.cmnp(images.gpu(), mirror=mirror)
        return [output, inputs["image/class/label"]]


class FileReaderPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        num_threads,
        device_id,
        data_dir,
        image_size=224,
        dali_device="cpu",
        rank=0,
        world_size=1,
        data_augmentation=True,
        shuffle=True,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        print(device_id)
        shard_id = rank
        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=shard_id,
            num_shards=world_size,
            random_shuffle=shuffle,
        )
        self.data_augmentation = data_augmentation
        if dali_device == "cpu":
            self.decode = ops.ImageDecoder(device=dali_device, output_type=types.RGB)
        else:
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoder(
                device="mixed",
                output_type=types.RGB,
                device_memory_padding=211025920,
                host_memory_padding=140544512,
            )

        if data_augmentation:
            self.res = ops.RandomResizedCrop(
                device=dali_device,
                size=(image_size, image_size),
                interp_type=types.INTERP_LINEAR,
                random_aspect_ratio=[0.75, 4.0 / 3.0],
                random_area=[0.08, 1.0],
                num_attempts=100,
            )

            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(image_size, image_size),
                image_type=types.RGB,
            )
            self.coin = ops.CoinFlip(probability=0.5)
        else:
            self.res = ops.Resize(device=dali_device, resize_shorter=image_size)
            self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(image_size, image_size),
                image_type=types.RGB,
            )

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        mirror = self.coin() if self.data_augmentation else None
        output = self.cmnp(images.gpu(), mirror=mirror)
        return (output, self.labels)


class DALIWrapper(object):
    def gen_wrapper(pipeline):
        for data in pipeline:
            input = data[0]["data"]
            target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            yield input, target
        pipeline.reset()

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.pipeline)


def build_dali_data_loader_from_image_folder(
    data_dir,
    batch_size,
    nb_examples_per_epoch,
    workers=5,
    _worker_init_fn=None,
    fp16=False,
    dali_device="gpu",
    world_size=1,
    local_rank=0,
    rank=0,
    image_size=224,
    data_augmentation=True,
    shuffle=True,
):
    pipeline = FileReaderPipeline(
        batch_size=batch_size,
        num_threads=workers,
        data_dir=data_dir,
        image_size=image_size,
        dali_device=dali_device,
        device_id=local_rank,
        rank=rank,
        world_size=world_size,
        data_augmentation=data_augmentation,
        shuffle=shuffle,
    )
    pipeline.build()
    size_per_node = nb_examples_per_epoch // world_size
    return DALIWrapper(DALIClassificationIterator(pipeline, size=size_per_node))


def build_dali_data_loader_from_tfrecords(
    tfrecord_files,
    idx_files,
    nb_examples_per_epoch,
    batch_size,
    workers=5,
    _worker_init_fn=None,
    fp16=False,
    dali_device="gpu",
    world_size=1,
    local_rank=0,
    rank=0,
    image_size=224,
    data_augmentation=True,
    shuffle=True,
    shuffle_buffer_size=8192,
    prefetch_queue_depth=1,
    shard=False,
):
    pipeline = TFRecordPipeline(
        tfrecord_files=tfrecord_files,
        idx_files=idx_files,
        batch_size=batch_size,
        num_threads=workers,
        device_id=local_rank,
        image_size=image_size,
        dali_device=dali_device,
        rank=rank,
        world_size=world_size,
        data_augmentation=data_augmentation,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_queue_depth=prefetch_queue_depth,
        shard=shard,
    )
    pipeline.build()
    size_per_node = nb_examples_per_epoch // world_size
    return DALIWrapper(DALIClassificationIterator(pipeline, size=size_per_node))
