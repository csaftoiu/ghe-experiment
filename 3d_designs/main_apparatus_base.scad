// 3D Printable Cylinder with Internal Chamber and Pane Slot
// Dimensions in millimeters

// Main dimensions
outer_diameter = 80;  // 9cm
outer_radius = outer_diameter / 2;
height = 10;         // 10cm

// Internal hole dimensions  
hole_diameter = 40;   // 4cm (changed from 30mm)
hole_radius = hole_diameter / 2;
hole_height = 5;      // 8cm (10cm - 2cm from bottom)


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
    difference() {
        // Main outer cylinder + more padding below 
        union() {
            cylinder(h = height - 2 + PANE_HOLE_HEIGHT_PADDING, r = outer_radius, center = false);
            translate([0, 0, -2]) {            
                cylinder(h = 2, r = outer_radius, center = false);
            }
        }
        
        // Main internal hole
        translate([0, 0, (height - hole_height)]) {
            cylinder(h = hole_height + 0.1, r = hole_radius, center = false);
        }
        
        // side slot, center hole 
        translate([0, 0, side_slot_ypos]) {
            cylinder(h = side_slot_height + PANE_HOLE_HEIGHT_PADDING, r = side_slot_width/2, center = false);
        }


        // wire path
        // away from the slot
        rotate([0, 0, 180]) {
        
            translate([3, -1, 1]) {        
                cube([60, 2, 2], center = false);
            }
           
            translate([2.5, -1, 1]) {
                cube([2, 2, 10], center=false);
            }        
        }
        // tiny little hole for thermocouple at the end 
        translate([-0.75, -0.75, hole_height-2.5]) {
            cube([1.5, 1.5, 1.5], center=false);
        }
        // little channel for the tc wire 
        translate([-3, -0.25, hole_height-1.5]) {
            cube([3, 0.5, 0.5], center=false);
        }        
        // 1cm x 1cm x 1mm lid on top 
        translate([-5, -5, hole_height-1]) {
            cube([10, 10, 1], center=false);
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

module cap() {
    translate([-5, -5, 13]) {
        // Main cube with 1mm triangular cutouts from each corner
        difference() {
            // Original cube
            cube([10, 10, 1], center=false);
            
            // Triangle cutouts at each corner
            // Bottom face corners
            translate([0, 0, 0]) 
                linear_extrude(height=1.1) 
                    polygon([[0,0], [1,0], [0,1]]);
            
            translate([10, 0, 0]) 
                linear_extrude(height=1.1) 
                    polygon([[0,0], [-1,0], [0,1]]);
            
            translate([10, 10, 0]) 
                linear_extrude(height=1.1) 
                    polygon([[0,0], [-1,0], [0,-1]]);
            
            translate([0, 10, 0]) 
                linear_extrude(height=1.1) 
                    polygon([[0,0], [1,0], [0,-1]]);
        }
    }
}

difference() {            
    top_part();
//    translate([-50, 0, -50]) cube([100, 100, 100], center=false);
}

difference() {            
    bot_part();
//    translate([-50, 0, -50]) cube([100, 100, 100], center=false);
}
cap();