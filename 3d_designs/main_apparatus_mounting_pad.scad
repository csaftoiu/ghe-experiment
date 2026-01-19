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

        linear_extrude(NEEDED_HEIGHT)
            arc(100, 63.5/2 - 4.826 - 2, 60-ARC_ANGLE_SCREWS/2, 60+ARC_ANGLE_SCREWS/2);
            
        linear_extrude(NEEDED_HEIGHT)
            arc(100, 63.5/2 - 4.826 - 2, 180-ARC_ANGLE_SCREWS/2, 180+ARC_ANGLE_SCREWS/2);

        linear_extrude(NEEDED_HEIGHT)
            arc(100, 63.5/2 - 4.826 - 2, 300-ARC_ANGLE_SCREWS/2, 300+ARC_ANGLE_SCREWS/2);
            
        linear_extrude(NEEDED_HEIGHT)
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

module pyrgepyra_padding() {
    // need extra height because its extra sensor PLUS padding
    union() {
    difference() {
        cylinder(h=NEEDED_HEIGHT, r=80/2);

        linear_extrude(NEEDED_HEIGHT)
            arc(100, 63.5/2 - 4.826 - 2, 60-ARC_ANGLE_SCREWS/2, 60+ARC_ANGLE_SCREWS/2);
            
        linear_extrude(NEEDED_HEIGHT)
            arc(100, 63.5/2 - 4.826 - 2, 180-ARC_ANGLE_SCREWS/2, 180+ARC_ANGLE_SCREWS/2);

        linear_extrude(NEEDED_HEIGHT)
            arc(100, 63.5/2 - 4.826 - 2, 300-ARC_ANGLE_SCREWS/2, 300+ARC_ANGLE_SCREWS/2);
            
        linear_extrude(NEEDED_HEIGHT)
            arc(100, 63.5/2 - 13.5, 120-ARC_ANGLE_BUBBLE/2, 120+ARC_ANGLE_BUBBLE/2);
            
        // space for cable
            // space for the cable  which is 5mm thick and goes up 8mm 
            // wire path
            // away from the slot
            rotate([0, 0, 0]) {
            
                translate([0, -8/2, 0]) {        
                    cube([60, 8, 10], center = false);
                }
            }
        
    }
        translate([0, 0, NEEDED_HEIGHT]) cylinder(h=7.9, r=80/2);
    }

    // screw holes for ref
    /*
    translate([cos(60)*63.5/2, sin(60)*63.5/2, 0]) cylinder(h=NEEDED_HEIGHT, r=4.826/2);
    translate([cos(180)*63.5/2, sin(180)*63.5/2, 0]) cylinder(h=NEEDED_HEIGHT, r=4.826/2);
    translate([cos(300)*63.5/2, sin(300)*63.5/2, 0]) cylinder(h=NEEDED_HEIGHT, r=4.826/2);
    translate([cos(120)*(63.5/2 - 11.5/2), sin(120)*(63.5/2 - 11.5/2), 0]) cylinder(h=NEEDED_HEIGHT, r=11.5/2);
    */
}




rotate([0,0,40]) tempmon_padding();
/*translate([100, 0, 0]) tempmon_padding();
translate([200, 0, 0]) tempmon_padding();
translate([0, 100, NEEDED_HEIGHT+7.9]) rotate([0,180,60]) pyrgepyra_padding();
translate([100, 100, NEEDED_HEIGHT+7.9]) rotate([0,180,60]) pyrgepyra_padding();
*/
