"""Adapted from https://github.com/ranandalon/mtl/blob/master/src/OPTICS.py

BSD 2-Clause License

Copyright (c) 2019, ranandalon
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import numpy as np


class Point():
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.cx = x + vx
        self.cy = y + vy
        self.cd = None  # core distance
        self.rd = None  # reachability distance
        self.processed = False


    def distance(self, point):
        return np.sqrt((point.cx - self.cx)**2 + (point.cy - self.cy)**2)


class Cluster:
    def __init__(self, points):
        self.points = points


    def centroid(self):
        center = [sum([p.cx for p in self.points]) / len(self.points),
                  sum([p.cy for p in self.points]) / len(self.points)]
        return center


class Optics():
    def __init__(self, pts_list, min_cluster_size, epsilon):
        self.pts = pts_list
        self.min_cluster_size = min_cluster_size
        self.max_radius = epsilon


    def _setup(self):
        for p in self.pts:
            p.rd = None
            p.processed = False
        self.unprocessed = [p for p in self.pts]
        self.ordered = []


    def _core_distance(self, point, neighbors):
        if point.cd is not None:
            return point.cd
        if len(neighbors) >= self.min_cluster_size - 1:
            sorted_neighbors = sorted([n.distance(point) for n in neighbors])
            point.cd = sorted_neighbors[self.min_cluster_size - 2]
        return point.cd


    def _neighbors(self, point):
        return [p for p in self.pts
                if (p is not point) and (p.distance(point) <= self.max_radius)]


    def _processed(self, point):
        point.processed = True
        self.unprocessed.remove(point)
        self.ordered.append(point)


    def _update(self, neighbors, point, seeds):
        for n in neighbors:
            if not n.processed:
                new_rd = max(point.cd, point.distance(n))
                if n.rd is None:
                    n.rd = new_rd
                    seeds.append(n)
                elif new_rd < n.rd:
                    n.rd = new_rd


    def run(self):
        self._setup()
        while self.unprocessed:
            point = self.unprocessed[0]
            self._processed(point)
            point_neighbors = self._neighbors(point)
            if self._core_distance(point, point_neighbors) is not None:
                seeds = []
                self._update(point_neighbors, point, seeds)
                while (seeds):
                    seeds.sort(key=lambda n: n.rd)
                    n = seeds.pop(0)
                    self._processed(n)
                    n_neighbors = self._neighbors(n)
                    if self._core_distance(n, n_neighbors) is not None:
                        self._update(n_neighbors, n, seeds)
        return self.ordered


    def cluster(self, cluster_threshold):
        clusters = []
        separators = []
        for i in range(len(self.ordered)):
            this_i = i
            next_i = i + 1
            this_p = self.ordered[i]
            if this_p.rd is not None:
                this_rd = this_p.rd
            else:
                this_rd = float('infinity')
            if this_rd > cluster_threshold:
                separators.append(this_i)
        separators.append(len(self.ordered))

        for i in range(len(separators) - 1):
            start = separators[i]
            end = separators[i + 1]
            if end - start >= self.min_cluster_size:
                clusters.append(Cluster(self.ordered[start:end]))
        return clusters
