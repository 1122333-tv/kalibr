"""CSV-backed EuRoC readers; timestamps remain integer nanoseconds."""
import csv
import re
from pathlib import Path

import aslam_cv as acv
import cv2
import numpy as np


def sensor_directory(root, sensor):
    if not re.fullmatch(r'(cam|imu)\d+', sensor):
        raise ValueError('Invalid EuRoC sensor directory: ' + sensor)
    root = Path(root)
    return (root if root.name == 'mav0' else root / 'mav0') / sensor


class _Dataset:
    def __init__(self, root, sensor, topic, from_to=None, freq=None):
        self.directory = sensor_directory(root, sensor)
        self.topic = topic
        self.csv_path = self.directory / 'data.csv'
        self.rows = []
        with self.csv_path.open(newline='') as stream:
            for line, row in enumerate(csv.reader(stream), 1):
                if not row or not row[0].strip() or row[0].lstrip().startswith('#'):
                    continue
                try:
                    stamp = int(row[0])
                    if stamp < 0 or (self.rows and stamp <= self.rows[-1][0]):
                        raise ValueError('timestamps must be nonnegative and strictly increasing')
                    value = self.parse_row(row)
                except (ValueError, IndexError) as exc:
                    raise ValueError('{}:{}: {}'.format(self.csv_path, line, exc))
                self.rows.append((stamp, value, line))
        if not self.rows:
            raise ValueError('{}: empty dataset'.format(self.csv_path))
        self.indices = list(range(len(self.rows)))
        if from_to is not None:
            start, end = from_to
            if not np.isfinite([start, end]).all() or start >= end:
                raise ValueError('Time interval must be finite with start < end')
            origin = self.rows[0][0]
            self.indices = [i for i in self.indices if start <= (self.rows[i][0]-origin)/1e9 <= end]
        if freq is not None:
            if not np.isfinite(freq) or freq <= 0:
                raise ValueError('Frequency must be positive and finite')
            selected = []
            for i in self.indices:
                # Match the bag reader\'s floating-point seconds frequency rule.
                t = self.rows[i][0] // 10**9 + self.rows[i][0] % 10**9 / 1e9
                if not selected or t - last >= 1.0 / freq:
                    selected.append(i)
                    last = t
            self.indices = selected
        if not self.indices:
            raise ValueError('{}: no samples in selected interval'.format(self.csv_path))

    def timestamp(self, index):
        return acv.Time(*divmod(self.rows[index][0], 10**9))

    def __iter__(self):
        return self.readDataset()

    def readDataset(self):
        return (self.get(i) for i in self.indices)

    def readDatasetShuffle(self):
        indices = list(self.indices)
        np.random.shuffle(indices)
        return (self.get(i) for i in indices)


class EurocImageDatasetReader(_Dataset):
    def __init__(self, root, sensor, topic, resolution, from_to=None, freq=None):
        self.resolution = tuple(resolution)
        super().__init__(root, sensor, topic, from_to, freq)

    def parse_row(self, row):
        if len(row) != 2:
            raise ValueError('Expected timestamp,filename')
        path = (self.directory / 'data' / row[1]).resolve()
        if not path.is_file():
            raise ValueError('Missing image: ' + str(path))
        return path

    def numImages(self):
        return len(self.indices)

    def getImage(self, index):
        _, path, line = self.rows[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[::-1] != self.resolution:
            raise ValueError('{}:{}: unreadable image or resolution mismatch: {}'.format(self.csv_path, line, path))
        return self.timestamp(index), image

    get = getImage


class EurocImuDatasetReader(_Dataset):
    def parse_row(self, row):
        if len(row) != 7:
            raise ValueError('Expected timestamp,wx,wy,wz,ax,ay,az')
        values = np.array([float(x) for x in row[1:]], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite IMU value')
        return values

    def numMessages(self):
        return len(self.indices)

    def getMessage(self, index):
        values = self.rows[index][1]
        return self.timestamp(index), values[:3].copy(), values[3:].copy()

    get = getMessage
