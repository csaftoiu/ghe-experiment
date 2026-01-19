// 3D Printable Cylinder with Internal Chamber and Pane Slot
// Dimensions in millimeters

// Main dimensions
outer_diameter = 80;  // 9cm
outer_radius = outer_diameter / 2;
height = 10;         // 10cm

// Internal hole dimensions  
hole_diameter = 40;   // 4cm (changed from 30mm)
hole_radius = hole_diameter / 2;
hole_height = 5;


PANE_HOLE_DIAMETER_PADDING = 0.8;  // make center hole a bit wider
PANE_HOLE_HEIGHT_PADDING = 0.4;    // make the bottom part a bit taller


// Side slot dimensions - UPDATED to go through to center
side_slot_height = 2;      // 2mm thick (tall)
side_slot_width = 50 + PANE_HOLE_DIAMETER_PADDING;      // 50mm wide (5cm) + padding
side_slot_ypos = 6;        // pos from the bottom 
side_slot_position = 55;   // 1mm above bottom of inner hole

// Rendering quality
$fn = 90;  // Smooth circles


detach_height = 30;       // 10cm separation between parts
split_point = side_slot_ypos; // Split at base of slot (6mm from bottom)


module top_part_main() {
    difference() {
        // Top cylinder from split point to top
        cylinder(h = height - split_point, r = outer_radius, center = false);
        
        // Continue internal hole in top part
        translate([0, 0, -(split_point - (height - hole_height))]) {
            cylinder(h = hole_height + 0.1, r = hole_radius, center = false);
        }
    }
}

module hollow_cylinder(h, ro, ri, center) {
    difference() {    
        cylinder(h=h, r=ro, center=center);
        cylinder(h=h, r=ri, center=center);
    }
}

JUTTERS_HEIGHT = 4;
JUTTERS_SHORTER_NESS = 0.4;
JUTTERS_THICKNESS = 1.6;
JUTTER_SPOT1 = 30;
JUTTER_SPOT2 = 35;
JUTTERS_RHOLE_PADDING = 0.4;

module top_part_jutters() {
    translate([0, 0, -JUTTERS_HEIGHT + JUTTERS_SHORTER_NESS]) {
        hollow_cylinder(JUTTERS_HEIGHT - JUTTERS_SHORTER_NESS, 
                        JUTTER_SPOT1 + JUTTERS_THICKNESS, JUTTER_SPOT1, center=false);
        hollow_cylinder(JUTTERS_HEIGHT - JUTTERS_SHORTER_NESS, 
                        JUTTER_SPOT2 + JUTTERS_THICKNESS, JUTTER_SPOT2, center=false);
    }
}


module top_part() {
    translate([0, 0, split_point + detach_height + 2 + PANE_HOLE_HEIGHT_PADDING]) {
        union() {        
            top_part_main();
            top_part_jutters();
        }
    }
}


module bot_part_main() {
    // up to where the smaller inner hole starts, we want the pyranometer 
    // base of the dome to be there (which is 27.92mm)
    // so we need 27.92mm total on the bottom
    // we already have 3, so extra needed is:            
    EXTRA_BOTTOM = 27.92 - 3;   

    difference() {
        // Main outer cylinder + more padding below 
        union() {
            cylinder(h = height - 2 + PANE_HOLE_HEIGHT_PADDING, r = outer_radius, center = false);
            
            translate([0, 0, -EXTRA_BOTTOM]) {            
                cylinder(h = EXTRA_BOTTOM, r = outer_radius, center = false);
            }
        }
        
        // Main internal hole: ADD 2mm 
        translate([0, 0, (height - hole_height - 2)]) {
            cylinder(h = hole_height, r = hole_radius, center = false);
        }
        
        // hole for the glass pane 
        translate([0, 0, side_slot_ypos]) {
            cylinder(h = side_slot_height + PANE_HOLE_HEIGHT_PADDING, r = side_slot_width/2, center = false);
        }
        
        // make a 23.5mm/2 + 0.4mm padding radius cylindrical hole all the way through 
        translate([0, 0, -40]) {
            cylinder(h = 80, r = 23.5/2 + 0.4, center = false);
        }
        
        // space for the cable  which is 5mm thick and goes up 8mm 
        // wire path
        // away from the slot
        rotate([0, 0, 180]) {
        
            translate([0, -8/2, -EXTRA_BOTTOM]) {        
                cube([60, 8, 10], center = false);
            }
        }
    }
}

module bot_part() {
    difference() {    
        bot_part_main();
        
        // take away the jutters inwards
        union() {
            // go facing downards startng from the top 
            rotate([0, 180, 0]) {
                translate([0, 0, -(8 + PANE_HOLE_HEIGHT_PADDING)]) {
                    hollow_cylinder(JUTTERS_HEIGHT, 
                                    JUTTER_SPOT1 + JUTTERS_THICKNESS + JUTTERS_RHOLE_PADDING, JUTTER_SPOT1 - JUTTERS_RHOLE_PADDING, center=false);
                    hollow_cylinder(JUTTERS_HEIGHT, 
                                    JUTTER_SPOT2 + JUTTERS_THICKNESS + JUTTERS_RHOLE_PADDING, JUTTER_SPOT2 - JUTTERS_RHOLE_PADDING, center=false);
                }            
            }
        }
    }
}



difference() {            
    top_part();
    translate([-50, 0, -50]) cube([100, 100, 100], center=false);
}

difference() {            
    bot_part();
    translate([-50, 0, -50]) cube([100, 100, 100], center=false);
}
