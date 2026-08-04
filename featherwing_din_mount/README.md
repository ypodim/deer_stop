# FeatherWing DIN Rail Mount

A parametric **DIN-rail mount for Adafruit Feather / FeatherWing** boards.

Based on [*Din Rail Mount* by Selphb (thing:1110765)](https://www.thingiverse.com/thing:1110765),
modified per request:

1. **Wider** front plate.
2. The original **three M3 holes** are replaced with **four holes in the Adafruit
   Feather corner mounting-hole pattern**, matching the corner holes of a
   FeatherWing such as [Adafruit #2923](https://www.adafruit.com/product/2923).

> The original Thingiverse STL is gated behind a Thingiverse account/API, so this
> is a clean parametric **reconstruction** in OpenSCAD rather than an edit of the
> original mesh. Every dimension is a tweakable parameter at the top of the file.

## Key dimensions

| Feature | Value |
|---|---|
| Rail | EN 50022 / TS35 "top hat", 35 mm wide |
| Front plate (perpendicular, default) | 31.78 × 59.72 × 3 mm (rounded corners, auto-sized) |
| Front plate (parallel) | 59.72 × 44 × 3 mm |
| Feather hole rectangle | 45.72 × 17.78 mm (1.8″ × 0.7″) |
| Holes | Ø2.1 mm self-tapping pilot for M2.5 (set `hole_d = 2.7` for clearance) |
| Standoffs | 4 mm tall, Ø6 mm (lifts the PCB off the plate) |

The four hole centers sit 2.54 mm in from each Feather PCB edge — the standard
Feather/Wing footprint, so any Feather-form-factor board lines up.

## Board orientation (`board_perpendicular`)

- `true` (default): board **long axis across the rail**. The plate auto-sizes to
  ~32 × 60 mm. The 45.72 mm hole pair lands at y = ±22.86 mm — **outboard of the
  35 mm rail and its hooks** — so the board screws exit into open air:
  - clears the rail edge by **5.36 mm**
  - clears the hook bracket by **2.66 mm** (≈1.3 mm even with Ø2.7 clearance holes)
- `false`: board **long axis along the rail** (original layout). Plate ~60 × 44 mm.
  Here the holes sit over the rail face, so keep screws ≤ ~6 mm so the tips don't
  reach the rail.

The plate is auto-sized from the hole pattern + `edge_margin`; edit `edge_margin`
to grow/shrink the border.

Pre-rendered: `featherwing_din_mount.stl` (perpendicular),
`featherwing_din_mount_parallel.stl` (parallel).

## Build the STL

```sh
openscad -o featherwing_din_mount.stl featherwing_din_mount.scad
```

Or open `featherwing_din_mount.scad` in the OpenSCAD GUI and press **F6**, then export.

## Print & assemble

- Print plate-face-down, 100% infill recommended (matches the original; the rail
  hooks take repeated stress).
- **Mount to rail:** hook the *top* lip over the upper rail edge, then rotate the
  body down until the *lower* hook snaps behind the lower edge (tilt-on).
- **Mount the board:** drop the FeatherWing onto the four standoffs and drive
  self-tapping M2.5 screws through the PCB into the standoffs from the top.

## Tuning

All parameters live at the top of `featherwing_din_mount.scad`:

- `plate_w` / `plate_h` — make it wider/taller still.
- `rail_clear` — loosen/tighten the rail grip if it's too tight or rattles.
- `catch_y` / `catch_z` — how aggressively the hooks bite the rail edges.
- `hole_d` — 2.1 (self-tap) vs 2.7 (clearance + nut).
- `standoffs` — set `false` to bolt the board flat against the plate.
