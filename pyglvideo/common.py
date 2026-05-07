import fractions
from dataclasses import dataclass

class FrameInfo(object):
    def __init__(self,pts:int,time_base:fractions.Fraction,duration:int=-1,key=0):
        self.pts=pts
        self.duration=duration
        self.time_base=time_base
        self.key=key

    @property
    def seconds(self):
        return float(self.pts*self.time_base)
    
    def __repr__(self):
        return f"FrameInfo@{self.seconds} | pts : {self.pts} | time_base : {self.time_base} | key : {self.key}"