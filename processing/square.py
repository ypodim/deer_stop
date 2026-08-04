"""
square.py — a simple red square in the center.

Run it:  ./processing/run.sh square.py
"""
import py5

SIZE = 120  # side length of the square


def setup():
    py5.size(600, 600)
    py5.rect_mode(py5.CENTER)   # x, y is the square's center, not its corner


def draw():
    py5.background(255)
    py5.no_stroke()
    py5.fill(255, 0, 0)         # red
    py5.square(py5.width / 2, py5.height / 2, SIZE)


py5.run_sketch()
