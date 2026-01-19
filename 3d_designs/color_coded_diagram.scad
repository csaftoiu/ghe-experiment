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


detach_height = 15;       // 10cm separation between parts
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
        translate([0,0,-0.01]) cylinder(h=h+0.02, r=ri, center=center);
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
            translate([0, 0, -30.82]) {            
                cylinder(h = 30.82, r = outer_radius, center = false);
            }
        }
        
        // Main internal hole
        translate([0, 0, (height - hole_height-0.01)]) {
            cylinder(h = hole_height + 0.1, r = hole_radius, center = false);
        }
        
        // side slot, center hole 
        translate([0, 0, side_slot_ypos]) {
            cylinder(h = side_slot_height + PANE_HOLE_HEIGHT_PADDING+0.01, r = side_slot_width/2, center = false);
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
        union() {
            color([0.5,0.5,1,1]) bot_part_main();
            // make black just display only thing for hole 
            difference() {color([0,0,0,1.0]) translate([0, 0, (height - hole_height-0.01)]) {
                cylinder(h=0.0001, r = hole_radius, center = false);    
            }
                    translate([-5, -5, hole_height-1]) {
            cube([10, 10, 1], center=false);
        }
            }
            
            // highlight the lip in bright color 
            translate([0,0,5]) color([1,0,0,1]) difference() {
                cylinder(h=1.01, r=25, center=false);
                translate([0,0,-0.1]) cylinder(h=2, r=19.99, center=false);
            }

        }
        
        // take away the jutters inwards
        union() {
            // go facing downards startng from the top 
            rotate([0, 180, 0]) {
                translate([0, 0, -(8 + PANE_HOLE_HEIGHT_PADDING+0.01)]) {
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
    translate([-5, -5, 7]) {
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

module extra_padding() {
SCREW_THREAD_SIZE = 19.05;
SCREW_HEAD_SIZE = 4.5;
SOLAR_PLATE_THICKNESS = 7.9;

NEEDED_HEIGHT = SCREW_THREAD_SIZE + SCREW_HEAD_SIZE - SOLAR_PLATE_THICKNESS + 2;

echo("total height padding needed:");
echo(NEEDED_HEIGHT);


$fn = 90;

module arc(r1, r2, a1, a2) {
  difference() {
    difference() {
      polygon([[0,0], [cos(a1) * (r1 + 50), sin(a1) * (r1 + 50)], [cos(a2) * (r1 + 50), sin(a2) * (r1 + 50)]]);
      circle(r = r2);
    }
    difference() {
      circle(r=r1 + 100);
      circle(r=r1);
    }
  }
}

// make a cylinder that is DIFF_NEEDED tall and 80mm diameter


ARC_ANGLE_SCREWS = 70; // 12.3;
ARC_ANGLE_BUBBLE = 29.8;

SCREW_DIAM_NEEDED = 40;  // 4.826

module tempmon_padding() {
    union () {
    difference() {
        cylinder(h=NEEDED_HEIGHT, r=80/2);

        translate([0,0,-0.05]) linear_extrude(NEEDED_HEIGHT+0.1)
            arc(100, 63.5/2 - 4.826 - 2, 60-ARC_ANGLE_SCREWS/2, 60+ARC_ANGLE_SCREWS/2);
            
        translate([0,0,-0.05]) linear_extrude(NEEDED_HEIGHT+0.1)
            arc(100, 63.5/2 - 4.826 - 2, 180-ARC_ANGLE_SCREWS/2, 180+ARC_ANGLE_SCREWS/2);

        translate([0,0,-0.05]) linear_extrude(NEEDED_HEIGHT+0.1)
            arc(100, 63.5/2 - 4.826 - 2, 300-ARC_ANGLE_SCREWS/2, 300+ARC_ANGLE_SCREWS/2);
            
        translate([0,0,-0.05]) linear_extrude(NEEDED_HEIGHT+0.1)
            arc(100, 63.5/2 - 13.5, 120-ARC_ANGLE_BUBBLE/2, 120+ARC_ANGLE_BUBBLE/2);
            
        
    }
    /*
    difference() {
        cylinder(3, 40, 40);
        cylinder(3, 37, 37);
    }*/
    
    }

    // screw holes for ref
    /*
    translate([cos(60)*63.5/2, sin(60)*63.5/2, 0]) cylinder(h=NEEDED_HEIGHT, r=SCREW_DIAM_NEEDED/2);
    translate([cos(180)*63.5/2, sin(180)*63.5/2, 0]) cylinder(h=NEEDED_HEIGHT, r=SCREW_DIAM_NEEDED/2);
    translate([cos(300)*63.5/2, sin(300)*63.5/2, 0]) cylinder(h=NEEDED_HEIGHT, r=SCREW_DIAM_NEEDED/2);
    translate([cos(120)*(63.5/2 - 11.5/2), sin(120)*(63.5/2 - 11.5/2), 0]) cylinder(h=NEEDED_HEIGHT, r=11.5/2);
    */
}



rotate([0,0,60]) translate([0,0,-50]) tempmon_padding();
/*translate([100, 0, 0]) tempmon_padding();
translate([200, 0, 0]) tempmon_padding();
translate([0, 100, NEEDED_HEIGHT+7.9]) rotate([0,180,60]) pyrgepyra_padding();
translate([100, 100, NEEDED_HEIGHT+7.9]) rotate([0,180,60]) pyrgepyra_padding();
*/


}

difference() {            
    color([1.0,1,0.0,1]) top_part();
    translate([-100, 0, -50]) cube([100, 100, 100], center=false);
}

difference() {            
    bot_part();
    translate([-100, 0, -50]) cube([100, 100, 100], center=false);
}
difference() {
    color([0.25,0.25,0.25,1.0]) cap();
    translate([-100, 0, -50]) cube([100, 100, 100], center=false);
}

difference() {
    color([1,0.5,0.5,1]) extra_padding();
//    translate([-50, 0, -100]) cube([100, 100, 200], center=false);
}


// the pane
translate([0, 0, 12]) color([0.4, 1, 1, 0.325]) cylinder(h=2, r=25, center=false);

// solar leveler 
color([0.6,0.6,0.6]) translate([0,0,-65]) cylinder(h=10, r=40, center=false);