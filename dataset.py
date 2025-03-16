import tensorflow_datasets as tfds

# โหลด dataset
dataset_name = "cats_vs_dogs"
dataset, info = tfds.load(dataset_name, as_supervised=True, with_info=True)

# บันทึกข้อมูลลงเครื่อง
tfds.benchmark(dataset)