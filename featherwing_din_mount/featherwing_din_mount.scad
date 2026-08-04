// ============================================================================
//  DIN Rail Mount for Adafruit Feather / FeatherWing
// ============================================================================
//  Parametric reconstruction of "Din Rail Mount" by Selphb
//  (Thingiverse thing:1110765), with two changes requested:
//
//    1. WIDER front plate.
//    2. The original THREE M3 holes are replaced with FOUR holes laid out in
//       the Adafruit Feather corner mounting-hole pattern, so an Adafruit
//       FeatherWing (e.g. product 2923, Latching Mini Relay FeatherWing) or any
//       Feather-form-factor board can be bolted directly to the rail mount.
//
//  Feather mounting-hole pattern (standard form factor, 0.9" x 2.0" board):
//    - 4 holes at the corners, 2.54 mm (0.1") nominal
//    - hole-center rectangle = 45.72 mm (1.8") x 17.78 mm (0.7")
//    - i.e. centers inset 2.54 mm (0.1") from each PCB edge
//
//  Rail: EN 50022 / TS35 "top hat", 35 mm wide, 7.5 mm deep, ~1.2 mm material.
//
//  Mounting: clip the top hook over the upper rail edge first, then rotate the
//  body down so the lower hook snaps behind the lower edge (tilt-on). Fasten
//  the FeatherWing to the four standoffs with self-tapping M2.5 screws driven
//  from the component side (screw passes through the PCB and bites the standoff).
// ============================================================================

/* [Board orientation] */
// false: board LONG axis runs ALONG the rail (parallel)
// true : board LONG axis runs ACROSS the rail (perpendicular). This pushes the
//        45.72 mm hole pair outboard of the rail edges, so the board screws
//        clear the rail/hooks entirely (more screw clearance, not less).
board_perpendicular = true;

/* [Front plate] */
plate_t   = 3;    // plate thickness
corner_r  = 3;    // rounded plate corners
edge_margin = 4;  // plate material around the outermost standoffs

/* [Feather corner hole pattern] */
feather_x = 45.72;  // long-axis hole-center spacing  (1.8")
feather_y = 17.78;  // short-axis hole-center spacing  (0.7")
hole_d    = 2.1;    // 2.1 = self-tapping pilot for M2.5. Use 2.7 for clearance.
standoffs   = true; // raise the board off the plate (clears bottom-side parts)
standoff_h  = 4;    // standoff height
standoff_d  = 6;    // standoff outer diameter

/* [DIN rail clip - EN 50022 / TS35] */
rail_w     = 35;    // rail width (outer edge to outer edge)
rail_clear = 0.4;   // total grip clearance across the rail
catch_z    = 1.6;   // how far behind the rail face the hook reaches
catch_y    = 2.2;   // how far the lip overlaps the rail edge (inward)
hook_thk   = 1.8;   // lip thickness (Z)
wall_thk   = 2.5;   // hook outer-wall thickness (Y)
hook_inset = 5;     // hook is shorter than the plate by this much each end

/* [Derived - do not edit] */
// hole-center spacing resolved onto the X (along-rail) and Y (across-rail) axes
hx = board_perpendicular ? feather_y : feather_x;   // along the rail
hy = board_perpendicular ? feather_x : feather_y;   // across the rail
// auto-size the plate: fit the standoffs + margin, and stay tall enough to host
// the rail hooks (which attach at y = +-(rail_w/2 + wall_thk))
min_h_for_hooks = rail_w + 2 * wall_thk + 2 * edge_margin;
plate_w = hx + standoff_d + 2 * edge_margin;
plate_h = max(min_h_for_hooks, hy + standoff_d + 2 * edge_margin);

$fn = 64;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
module at_feather_holes() {
    for (sx = [-1, 1], sy = [-1, 1])
        translate([sx * hx / 2, sy * hy / 2, 0]) children();
}

// rounded rectangular plate in XY, base at z = 0
module rounded_plate(w, h, t, r) {
    linear_extrude(t)
        offset(r) offset(-r)
            square([w, h], center = true);
}

// One rail hook for the +Y (top) edge, extruded along X.
// Mirror in Y for the bottom edge.
//   - outer wall drops straight back from the plate
//   - lip turns inward at the back to trap the rolled rail edge
module rail_hook() {
    y_outer = rail_w / 2 + rail_clear / 2;     // inner face of outer wall
    len     = plate_w - 2 * hook_inset;
    union() {
        // outer wall (behind the plate, -Z)
        translate([-len / 2, y_outer, -(catch_z + hook_thk)])
            cube([len, wall_thk, catch_z + hook_thk]);
        // inward catch lip at the very back
        translate([-len / 2, y_outer - catch_y, -(catch_z + hook_thk)])
            cube([len, catch_y + wall_thk, hook_thk]);
    }
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------
module featherwing_din_mount() {
    difference() {
        union() {
            // front plate
            translate([0, 0, 0]) rounded_plate(plate_w, plate_h, plate_t, corner_r);

            // board standoffs (front side)
            if (standoffs)
                at_feather_holes()
                    translate([0, 0, plate_t])
                        cylinder(h = standoff_h, d = standoff_d);

            // DIN rail hooks (back side), top + bottom
            rail_hook();
            mirror([0, 1, 0]) rail_hook();
        }

        // four Feather mounting holes, drilled through plate (+ standoff)
        at_feather_holes()
            translate([0, 0, -1])
                cylinder(h = plate_t + (standoffs ? standoff_h : 0) + 2, d = hole_d);
    }
}

featherwing_din_mount();
