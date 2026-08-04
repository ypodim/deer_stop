"""
flow.py — a Perlin-noise flow field to experiment with.

Run it:   ./processing/run.sh flow.py
Controls: move the mouse to bend the field, click to reseed,
          press 'c' to clear, 's' to save a frame.
"""
import py5

N = 1500            # number of particles
SCALE = 0.004       # noise frequency (smaller = smoother swirls)
SPEED = 1.6         # particle step length
particles = []
seed = 0.0


def reseed():
    global seed
    seed = py5.random(1000)
    py5.background(8, 10, 16)


def setup():
    py5.size(900, 900)
    py5.color_mode(py5.HSB, 360, 100, 100, 100)
    for _ in range(N):
        particles.append([py5.random(py5.width), py5.random(py5.height)])
    reseed()


def draw():
    # mouse pulls the field's "time" axis, so the whole flow morphs as you move
    mt = py5.mouse_x * 0.002

    py5.no_stroke()
    for p in particles:
        x, y = p
        angle = py5.noise(x * SCALE, y * SCALE, seed + mt) * py5.TWO_PI * 2
        x += py5.cos(angle) * SPEED
        y += py5.sin(angle) * SPEED

        # wrap around the edges
        if x < 0: x += py5.width
        if x > py5.width: x -= py5.width
        if y < 0: y += py5.height
        if y > py5.height: y -= py5.height

        hue = (angle * 30 + py5.frame_count * 0.2) % 360
        py5.fill(hue, 70, 100, 14)
        py5.circle(x, y, 1.8)
        p[0], p[1] = x, y


def mouse_pressed():
    reseed()


def key_pressed():
    if py5.key == 'c':
        py5.background(8, 10, 16)
    elif py5.key == 's':
        py5.save_frame("frame-####.png")


py5.run_sketch()
