import json

import jsonpickle
import numpy as np

from src.kinematics.kinematics_utils import Pose
from src.utils.movement import SplineMovement

arr = np.array([1.123, 1.123543], dtype=np.float64)
print(arr[0])
pose1 = Pose(1, arr[0], arr[0], alpha=arr[1], beta=arr[1], gamma=arr[1])
pose2 = Pose(2, arr[0], arr[0], alpha=arr[1], beta=arr[1], gamma=arr[1])
pose3 = Pose(3, arr[0], arr[0], alpha=arr[1], beta=arr[1], gamma=arr[1])

poses1 = [pose1, pose2]
poses2 = [pose1, pose2, pose3]

test1 = SplineMovement(None, poses1, 2)
test2 = SplineMovement(None, poses2, 3)

movements = [test1, test2]

json_string = jsonpickle.encode(movements)

print(json.dumps(json.loads(json_string), indent=4))

with open('test.json', 'w') as outfile:
    json.dump(json.loads(json_string), outfile, indent=4)

with open('test.json', 'r') as infile:
    string = infile.read()

loaded_data = jsonpickle.decode(string)
print(loaded_data)
