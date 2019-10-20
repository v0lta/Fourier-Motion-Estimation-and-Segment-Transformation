import sys
sys.path.insert(0, "../../")
from nowcasting.movingmnist_iterator import MovingMNISTAdvancedIterator
from networks.ops import preprocess_data
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="generate dataset")
parser.add_argument("-sl", "--seqlen", help="length of the total sequence",
                    type=int, default=32)
parser.add_argument("--digit_num", help="number of digits",
                    type=int, default=3)
parser.add_argument("--num", help="number of instances",
                    type=int, default=20)
args = parser.parse_args()
seqlen = args.seqlen
digit_num = args.digit_num
test_total = args.num
ROTATION = (-30, 30)
PATH = str(seqlen) + '-testdata'
mnt_generator = MovingMNISTAdvancedIterator(digit_num=digit_num,
                                            rotation_angle_range=ROTATION)
test_vid, _ = mnt_generator.sample(batch_size=test_total, seqlen=seqlen)
test_vid = np.clip(test_vid, 0, 255)
print("maximum value ", np.amax(test_vid))
print("minimum value ", np.amin(test_vid))
test_vid = preprocess_data(test_vid)
print(np.amax(test_vid))
print(np.amin(test_vid))
np.save(PATH, test_vid)
print(test_vid.shape)
print("digit_num ", digit_num)
print("instances ", test_total)
