CUBE_HEIGHT = 10;
JUTTER_HEIGHT = 43;
CUBE_SIDE = 43;


module pixelGeneric(x, y, xt, yt) {
    translate([x, y, CUBE_HEIGHT]) {
        cube([xt, yt, JUTTER_HEIGHT]);
    }
}


module pixelL(x, y) {
    pixelGeneric(x, y, 0.4, 1);
}
module pixelR(x, y) {
    pixelGeneric(x + 0.6, y, 0.4, 1);
}
module pixelB(x, y) {
    pixelGeneric(x, y + 0.6, 1, 0.4);
}
module pixelT(x, y) {
    pixelGeneric(x, y, 1, 0.4);
}
module pixelCR1(x, y) {
    pixelL(x, y);
    pixelT(x, y);
}
module pixelCR2(x, y) {
    pixelR(x, y);
    pixelT(x, y);
}
module pixelCR3(x, y) {
    pixelL(x, y);
    pixelB(x, y);
}
module pixelCR4(x, y) {
    pixelR(x, y);
    pixelB(x, y);
}

module crossSlit1(x, y) {
    pixelGeneric(x+0.3,y-1,0.4,2);
}
module crossSlit2(x, y) {
    pixelGeneric(x-1,y+0.3,2,0.4);
}
module crossSlit3(x, y) {
    pixelGeneric(x,y+0.3,2,0.4);
}
module crossSlit4(x, y) {
    pixelGeneric(x+0.3,y,0.4,2);
}

$fn = 90;

module fullPiece() {
union () {

cube([CUBE_SIDE, CUBE_SIDE, CUBE_HEIGHT]);

pixelL(0, 1);
pixelL(0, 2);
pixelL(0, 3);
pixelL(0, 4);
pixelL(2, 4);
pixelL(2, 5);
pixelL(4, 5);
pixelL(2, 6);
pixelL(2, 7);
pixelL(6, 7);
pixelL(0, 8);
pixelL(2, 8);
pixelL(4, 8);
pixelL(6, 8);
pixelL(0, 9);
pixelL(2, 9);
pixelL(4, 9);
pixelL(6, 9);
pixelL(8, 9);
pixelL(0, 10);
pixelL(4, 10);
pixelL(8, 10);
pixelL(0, 11);
pixelL(4, 11);
pixelL(8, 11);
pixelL(0, 12);
pixelL(4, 12);
pixelL(8, 12);
pixelL(0, 13);
pixelL(2, 13);
pixelL(4, 13);
pixelL(6, 13);
pixelL(8, 13);
pixelL(10, 13);
pixelL(0, 14);
pixelL(2, 14);
pixelL(4, 14);
pixelL(6, 14);
pixelL(8, 14);
pixelL(10, 14);
pixelL(2, 15);
pixelL(6, 15);
pixelL(10, 15);
pixelL(2, 16);
pixelL(6, 16);
pixelL(10, 16);
pixelL(2, 17);
pixelL(6, 17);
pixelL(10, 17);
pixelL(0, 18);
pixelL(2, 18);
pixelL(4, 18);
pixelL(6, 18);
pixelL(8, 18);
pixelL(10, 18);
pixelL(0, 19);
pixelL(2, 19);
pixelL(4, 19);
pixelL(6, 19);
pixelL(8, 19);
pixelL(10, 19);
pixelL(0, 20);
pixelL(4, 20);
pixelL(8, 20);
pixelL(0, 21);
pixelL(4, 21);
pixelL(8, 21);
pixelL(0, 22);
pixelL(4, 22);
pixelL(8, 22);
pixelL(0, 23);
pixelL(2, 23);
pixelL(4, 23);
pixelL(6, 23);
pixelL(8, 23);
pixelL(10, 23);
pixelL(0, 24);
pixelL(2, 24);
pixelL(4, 24);
pixelL(6, 24);
pixelL(8, 24);
pixelL(10, 24);
pixelL(2, 25);
pixelL(6, 25);
pixelL(10, 25);
pixelL(2, 26);
pixelL(6, 26);
pixelL(10, 26);
pixelL(2, 27);
pixelL(6, 27);
pixelL(10, 27);
pixelL(0, 28);
pixelL(2, 28);
pixelL(4, 28);
pixelL(6, 28);
pixelL(8, 28);
pixelL(10, 28);
pixelL(0, 29);
pixelL(2, 29);
pixelL(4, 29);
pixelL(6, 29);
pixelL(8, 29);
pixelL(10, 29);
pixelL(0, 30);
pixelL(4, 30);
pixelL(8, 30);
pixelL(0, 31);
pixelL(4, 31);
pixelL(8, 31);
pixelL(0, 32);
pixelL(4, 32);
pixelL(8, 32);
pixelL(0, 33);
pixelL(2, 33);
pixelL(4, 33);
pixelL(6, 33);
pixelL(8, 33);
pixelL(0, 34);
pixelL(2, 34);
pixelL(4, 34);
pixelL(6, 34);
pixelL(2, 35);
pixelL(6, 35);
pixelL(2, 36);
pixelL(2, 37);
pixelL(4, 37);
pixelL(0, 38);
pixelL(2, 38);
pixelL(0, 39);
pixelL(0, 40);
pixelL(0, 41);
pixelT(1, 0);
pixelT(2, 0);
pixelT(3, 0);
pixelT(4, 0);
pixelT(8, 0);
pixelT(9, 0);
pixelT(10, 0);
pixelT(11, 0);
pixelT(12, 0);
pixelT(13, 0);
pixelT(14, 0);
pixelT(18, 0);
pixelT(19, 0);
pixelT(20, 0);
pixelT(21, 0);
pixelT(22, 0);
pixelT(23, 0);
pixelT(24, 0);
pixelT(28, 0);
pixelT(29, 0);
pixelT(30, 0);
pixelT(31, 0);
pixelT(32, 0);
pixelT(33, 0);
pixelT(34, 0);
pixelT(38, 0);
pixelT(39, 0);
pixelT(40, 0);
pixelT(41, 0);
pixelT(4, 2);
pixelT(5, 2);
pixelT(6, 2);
pixelT(7, 2);
pixelT(8, 2);
pixelT(9, 2);
pixelT(13, 2);
pixelT(14, 2);
pixelT(15, 2);
pixelT(16, 2);
pixelT(17, 2);
pixelT(18, 2);
pixelT(19, 2);
pixelT(23, 2);
pixelT(24, 2);
pixelT(25, 2);
pixelT(26, 2);
pixelT(27, 2);
pixelT(28, 2);
pixelT(29, 2);
pixelT(33, 2);
pixelT(34, 2);
pixelT(35, 2);
pixelT(36, 2);
pixelT(37, 2);
pixelT(38, 2);
pixelT(5, 4);
pixelT(8, 4);
pixelT(9, 4);
pixelT(10, 4);
pixelT(11, 4);
pixelT(12, 4);
pixelT(13, 4);
pixelT(14, 4);
pixelT(18, 4);
pixelT(19, 4);
pixelT(20, 4);
pixelT(21, 4);
pixelT(22, 4);
pixelT(23, 4);
pixelT(24, 4);
pixelT(28, 4);
pixelT(29, 4);
pixelT(30, 4);
pixelT(31, 4);
pixelT(32, 4);
pixelT(33, 4);
pixelT(34, 4);
pixelT(37, 4);
pixelT(7, 6);
pixelT(8, 6);
pixelT(9, 6);
pixelT(13, 6);
pixelT(14, 6);
pixelT(15, 6);
pixelT(16, 6);
pixelT(17, 6);
pixelT(18, 6);
pixelT(19, 6);
pixelT(23, 6);
pixelT(24, 6);
pixelT(25, 6);
pixelT(26, 6);
pixelT(27, 6);
pixelT(28, 6);
pixelT(29, 6);
pixelT(33, 6);
pixelT(34, 6);
pixelT(35, 6);
pixelT(9, 8);
pixelT(10, 8);
pixelT(11, 8);
pixelT(12, 8);
pixelT(13, 8);
pixelT(14, 8);
pixelT(18, 8);
pixelT(19, 8);
pixelT(20, 8);
pixelT(21, 8);
pixelT(22, 8);
pixelT(23, 8);
pixelT(24, 8);
pixelT(28, 8);
pixelT(29, 8);
pixelT(30, 8);
pixelT(31, 8);
pixelT(32, 8);
pixelT(33, 8);
pixelT(13, 10);
pixelT(14, 10);
pixelT(15, 10);
pixelT(16, 10);
pixelT(17, 10);
pixelT(18, 10);
pixelT(19, 10);
pixelT(23, 10);
pixelT(24, 10);
pixelT(25, 10);
pixelT(26, 10);
pixelT(27, 10);
pixelT(28, 10);
pixelT(29, 10);
pixelR(42, 1);
pixelR(42, 2);
pixelR(42, 3);
pixelR(40, 4);
pixelR(42, 4);
pixelR(38, 5);
pixelR(40, 5);
pixelR(40, 6);
pixelR(36, 7);
pixelR(40, 7);
pixelR(36, 8);
pixelR(38, 8);
pixelR(40, 8);
pixelR(42, 8);
pixelR(34, 9);
pixelR(36, 9);
pixelR(38, 9);
pixelR(40, 9);
pixelR(42, 9);
pixelR(34, 10);
pixelR(38, 10);
pixelR(42, 10);
pixelR(34, 11);
pixelR(38, 11);
pixelR(42, 11);
pixelR(34, 12);
pixelR(38, 12);
pixelR(42, 12);
pixelR(32, 13);
pixelR(34, 13);
pixelR(36, 13);
pixelR(38, 13);
pixelR(40, 13);
pixelR(42, 13);
pixelR(32, 14);
pixelR(34, 14);
pixelR(36, 14);
pixelR(38, 14);
pixelR(40, 14);
pixelR(42, 14);
pixelR(32, 15);
pixelR(36, 15);
pixelR(40, 15);
pixelR(32, 16);
pixelR(36, 16);
pixelR(40, 16);
pixelR(32, 17);
pixelR(36, 17);
pixelR(40, 17);
pixelR(32, 18);
pixelR(34, 18);
pixelR(36, 18);
pixelR(38, 18);
pixelR(40, 18);
pixelR(42, 18);
pixelR(32, 19);
pixelR(34, 19);
pixelR(36, 19);
pixelR(38, 19);
pixelR(40, 19);
pixelR(42, 19);
pixelR(34, 20);
pixelR(38, 20);
pixelR(42, 20);
pixelR(34, 21);
pixelR(38, 21);
pixelR(42, 21);
pixelR(34, 22);
pixelR(38, 22);
pixelR(42, 22);
pixelR(32, 23);
pixelR(34, 23);
pixelR(36, 23);
pixelR(38, 23);
pixelR(40, 23);
pixelR(42, 23);
pixelR(32, 24);
pixelR(34, 24);
pixelR(36, 24);
pixelR(38, 24);
pixelR(40, 24);
pixelR(42, 24);
pixelR(32, 25);
pixelR(36, 25);
pixelR(40, 25);
pixelR(32, 26);
pixelR(36, 26);
pixelR(40, 26);
pixelR(32, 27);
pixelR(36, 27);
pixelR(40, 27);
pixelR(32, 28);
pixelR(34, 28);
pixelR(36, 28);
pixelR(38, 28);
pixelR(40, 28);
pixelR(42, 28);
pixelR(32, 29);
pixelR(34, 29);
pixelR(36, 29);
pixelR(38, 29);
pixelR(40, 29);
pixelR(42, 29);
pixelR(34, 30);
pixelR(38, 30);
pixelR(42, 30);
pixelR(34, 31);
pixelR(38, 31);
pixelR(42, 31);
pixelR(34, 32);
pixelR(38, 32);
pixelR(42, 32);
pixelR(34, 33);
pixelR(36, 33);
pixelR(38, 33);
pixelR(40, 33);
pixelR(42, 33);
pixelR(36, 34);
pixelR(38, 34);
pixelR(40, 34);
pixelR(42, 34);
pixelR(36, 35);
pixelR(40, 35);
pixelR(40, 36);
pixelR(38, 37);
pixelR(40, 37);
pixelR(40, 38);
pixelR(42, 38);
pixelR(42, 39);
pixelR(42, 40);
pixelR(42, 41);
pixelB(13, 32);
pixelB(14, 32);
pixelB(15, 32);
pixelB(16, 32);
pixelB(17, 32);
pixelB(18, 32);
pixelB(19, 32);
pixelB(23, 32);
pixelB(24, 32);
pixelB(25, 32);
pixelB(26, 32);
pixelB(27, 32);
pixelB(28, 32);
pixelB(29, 32);
pixelB(9, 34);
pixelB(10, 34);
pixelB(11, 34);
pixelB(12, 34);
pixelB(13, 34);
pixelB(14, 34);
pixelB(18, 34);
pixelB(19, 34);
pixelB(20, 34);
pixelB(21, 34);
pixelB(22, 34);
pixelB(23, 34);
pixelB(24, 34);
pixelB(28, 34);
pixelB(29, 34);
pixelB(30, 34);
pixelB(31, 34);
pixelB(32, 34);
pixelB(33, 34);
pixelB(7, 36);
pixelB(8, 36);
pixelB(9, 36);
pixelB(13, 36);
pixelB(14, 36);
pixelB(15, 36);
pixelB(16, 36);
pixelB(17, 36);
pixelB(18, 36);
pixelB(19, 36);
pixelB(23, 36);
pixelB(24, 36);
pixelB(25, 36);
pixelB(26, 36);
pixelB(27, 36);
pixelB(28, 36);
pixelB(29, 36);
pixelB(33, 36);
pixelB(34, 36);
pixelB(35, 36);
pixelB(5, 38);
pixelB(8, 38);
pixelB(9, 38);
pixelB(10, 38);
pixelB(11, 38);
pixelB(12, 38);
pixelB(13, 38);
pixelB(14, 38);
pixelB(18, 38);
pixelB(19, 38);
pixelB(20, 38);
pixelB(21, 38);
pixelB(22, 38);
pixelB(23, 38);
pixelB(24, 38);
pixelB(28, 38);
pixelB(29, 38);
pixelB(30, 38);
pixelB(31, 38);
pixelB(32, 38);
pixelB(33, 38);
pixelB(34, 38);
pixelB(37, 38);
pixelB(4, 40);
pixelB(5, 40);
pixelB(6, 40);
pixelB(7, 40);
pixelB(8, 40);
pixelB(9, 40);
pixelB(13, 40);
pixelB(14, 40);
pixelB(15, 40);
pixelB(16, 40);
pixelB(17, 40);
pixelB(18, 40);
pixelB(19, 40);
pixelB(23, 40);
pixelB(24, 40);
pixelB(25, 40);
pixelB(26, 40);
pixelB(27, 40);
pixelB(28, 40);
pixelB(29, 40);
pixelB(33, 40);
pixelB(34, 40);
pixelB(35, 40);
pixelB(36, 40);
pixelB(37, 40);
pixelB(38, 40);
pixelB(1, 42);
pixelB(2, 42);
pixelB(3, 42);
pixelB(4, 42);
pixelB(8, 42);
pixelB(9, 42);
pixelB(10, 42);
pixelB(11, 42);
pixelB(12, 42);
pixelB(13, 42);
pixelB(14, 42);
pixelB(18, 42);
pixelB(19, 42);
pixelB(20, 42);
pixelB(21, 42);
pixelB(22, 42);
pixelB(23, 42);
pixelB(24, 42);
pixelB(28, 42);
pixelB(29, 42);
pixelB(30, 42);
pixelB(31, 42);
pixelB(32, 42);
pixelB(33, 42);
pixelB(34, 42);
pixelB(38, 42);
pixelB(39, 42);
pixelB(40, 42);
pixelB(41, 42);

pixelCR1(0, 0);
pixelCR1(2, 2);
pixelCR1(4, 4);
pixelCR1(6, 6);
pixelCR1(8, 8);
pixelCR1(10, 10);

pixelCR2(42, 0);
pixelCR2(40, 2);
pixelCR2(38, 4);
pixelCR2(36, 6);
pixelCR2(34, 8);
pixelCR2(32, 10);

pixelCR3(0, 42);
pixelCR3(2, 40);
pixelCR3(4, 38);
pixelCR3(6, 36);
pixelCR3(8, 34);
pixelCR3(10, 32);

pixelCR4(32, 32);
pixelCR4(34, 34);
pixelCR4(36, 36);
pixelCR4(38, 38);
pixelCR4(40, 40);
pixelCR4(42, 42);

crossSlit1(6, 0);
crossSlit1(16, 0);
crossSlit1(26, 0);
crossSlit1(36, 0);
crossSlit1(11, 2);
crossSlit1(21, 2);
crossSlit1(31, 2);
crossSlit1(16, 4);
crossSlit1(26, 4);
crossSlit1(16, 8);
crossSlit1(26, 8);
crossSlit1(21, 10);
crossSlit1(11, 6);
crossSlit1(21, 6);
crossSlit1(31, 6);

crossSlit2(0, 6);
crossSlit2(2, 11);
crossSlit2(6, 11);
crossSlit2(0, 16);
crossSlit2(4, 16);
crossSlit2(8, 16);
crossSlit2(2, 21);
crossSlit2(6, 21);
crossSlit2(10, 21);
crossSlit2(0, 26);
crossSlit2(4, 26);
crossSlit2(8, 26);
crossSlit2(2, 31);
crossSlit2(6, 31);
crossSlit2(0, 36);

crossSlit3(36, 11);
crossSlit3(40, 11);
crossSlit3(34, 16);
crossSlit3(38, 16);
crossSlit3(42, 16);
crossSlit3(32, 21);
crossSlit3(36, 21);
crossSlit3(40, 21);
crossSlit3(34, 26);
crossSlit3(38, 26);
crossSlit3(42, 26);
crossSlit3(36, 31);
crossSlit3(40, 31);
crossSlit3(42, 6);

crossSlit4(21, 32);
crossSlit4(16, 34);
crossSlit4(26, 34);
crossSlit4(11, 36);
crossSlit4(21, 36);
crossSlit4(31, 36);
crossSlit3(42, 36);
crossSlit4(11, 40);
crossSlit4(21, 40);
crossSlit4(31, 40);
crossSlit4(6, 42);
crossSlit4(16, 42);
crossSlit4(26, 42);
crossSlit4(36, 42);
crossSlit4(16, 38);
crossSlit4(26, 38);

}
}

module prism(l, w, h) {
    polyhedron(// pt      0        1        2        3        4        5
               points=[[0,0,0], [0,w,h], [l,w,h], [l,0,0], [0,w,0], [l,w,0]],
               // top sloping face (A)
               faces=[[0,1,2,3],
               // vertical rectangular face (B)
               [2,1,4,5],
               // bottom face (C)
               [0,3,5,4],
               // rear triangular face (D)
               [0,4,1],
               // front triangular face (E)
               [3,2,5]]
               );}

module housing() {
difference() {
union() {
difference() {
    fullPiece();
    
    // holes for ventilation on top
//    translate([12, 12, 5-1]) cube([43-12*2, 2, 5+1]);
    translate([12, 12, 0]) cube([43-12*2, 0.4, 10]);
    translate([12, 14, 0]) cube([43-12*2, 0.4, 10]);
    translate([12, 16, 0]) cube([43-12*2, 0.4, 10]);
    translate([12, 43-12-0.4, 0]) cube([43-12*2, 0.4, 10]);
    translate([12, 43-14-0.4, 0]) cube([43-12*2, 0.4, 10]);
    translate([12, 43-16-0.4, 0]) cube([43-12*2, 0.4, 10]);
  //  translate([12, 12+2, 5-1]) cube([43-12*2, 2, 2]);
  
    // slope upwards into the vent slot 
    translate([12, 13, 10]) {
        rotate([0,180,180]) {
        // cube([43-12*2, 1, 2]);
        translate([0,-8.5,0]) prism(43-12*2, 8.5+1, 2);
        }
        rotate([0,180,180]) {
        // cube([43-12*2, 1, 2]);
        translate([0,-8.5,0]) rotate([0,0,180]) translate([-(43-12*2),0,0]) prism(43-12*2, 8.5+1, 2);
        }
    }
    
    // wire hole
    // 2x2 hole on top
    translate([CUBE_SIDE/2-5/2, CUBE_SIDE/2-5/2, 4]) cube([5, 5, 6]);
    // and wire path
    translate([CUBE_SIDE/2-2/2, CUBE_SIDE/2-2/2, 5-2/2]) cube([CUBE_SIDE, 2, 2]);

    // wire hole prism
    // slope upwards into the vent slot 
    translate([CUBE_SIDE/2-5/2, CUBE_SIDE/2-5/2, 4]) {
        rotate([0,180,180]) {
        // cube([43-12*2, 1, 2]);
        translate([0,-2.5,0]) prism(5, 2.5, 1);
        }
        rotate([0,180,180]) {
        // cube([43-12*2, 1, 2]);
        translate([0,-2.5,0]) rotate([0,0,180]) translate([-5,0,0]) prism(5, 2.5, 1);
        }
    }

    // wire hole vents
    translate([CUBE_SIDE/2 - 2.5, CUBE_SIDE/2 + 2.5 - 0.4, 0]) cube([5, 0.4, 20]);
    translate([CUBE_SIDE/2 - 2.5, CUBE_SIDE/2 - 2.5, 0]) cube([5, 0.4, 20]);
    
    // holes to mount the top 
    translate([8 - 2/2, 8 - 2/2, 0]) cube([2, 2, 5]);
    translate([CUBE_SIDE - 8 - 2/2, 8 - 2/2, 0]) cube([2, 2, 5]);
    translate([8 - 2/2, CUBE_SIDE - 8 - 2/2, 0]) cube([2, 2, 5]);
    translate([CUBE_SIDE - 8 - 2/2, CUBE_SIDE - 8 - 2/2, 0]) cube([2, 2, 5]);
}

// add pole at the end 
// middle pole
translate([CUBE_SIDE/2, CUBE_SIDE/2, 0]) {
    difference() {
        cylinder(CUBE_HEIGHT + JUTTER_HEIGHT/2-2, 1.5, 1.5);
//        translate([-0.5, -1, 3]) cube([2, 2, JUTTER_HEIGHT]);
        translate([1, 0, 0]) cylinder(CUBE_HEIGHT + JUTTER_HEIGHT/2-2, 1, 1);
    }
}

}
}

    // view slicer
//    translate([-5, -5, -5]) cube([CUBE_SIDE/2+5, 100, 100]);
//    translate([CUBE_SIDE/2, -5, 0]) cube([CUBE_SIDE/2+5,100,100]);
//    translate([-5, CUBE_SIDE/2, -5]) cube([100, 100, 100]);
}


// separate top piece
module cap() {
translate([0, 0, -30]) {
    union() {
        // cut out prism sides
        difference() {        
            cube([CUBE_SIDE, CUBE_SIDE, 5]);
            rotate([0, 180, 0]) translate([-CUBE_SIDE, CUBE_SIDE/2, -5]) prism(CUBE_SIDE, CUBE_SIDE/2, 3);
            rotate([0, 180, 0]) translate([-CUBE_SIDE, CUBE_SIDE/2, -5]) rotate([0, 0, 180]) translate([-CUBE_SIDE, 0, 0]) prism(CUBE_SIDE, CUBE_SIDE/2, 3);
        }
        
        // and add the poles 
        translate([8 - 2/2, 8 - 2/2, 0]) cube([2, 2, 25]);
        translate([CUBE_SIDE - 8 - 2/2, 8 - 2/2, 0]) cube([2, 2, 25]);
        translate([8 - 2/2, CUBE_SIDE - 8 - 2/2, 0]) cube([2, 2, 25]);
        translate([CUBE_SIDE - 8 - 2/2, CUBE_SIDE - 8 - 2/2, 0]) cube([2, 2, 25]);
        
        // make everything but the end thicker 
        translate([8 - 5/2, 8 - 5/2, 0]) cube([5, 5, 20]);
        translate([CUBE_SIDE - 8 - 5/2, 8 - 5/2, 0]) cube([5, 5, 20]);
        translate([8 - 5/2, CUBE_SIDE - 8 - 5/2, 0]) cube([5, 5, 20]);
        translate([CUBE_SIDE - 8 - 5/2, CUBE_SIDE - 8 - 5/2, 0]) cube([5, 5, 20]);
        
        // and add side shielding just in case 
        translate([8, 8 - 5/2, 0]) cube([CUBE_SIDE - 8*2, 1, 20]);
        translate([8, CUBE_SIDE - 8 + 5/2 - 1, 0]) cube([CUBE_SIDE - 8*2, 1, 20]);
        
    }
}
}

module spacer(spacerh) {
        // Cylinder with offset hole
        // Main cylinder: 2.73mm thick, 23.50mm radius
        // Hole: 5.826mm diameter, centered on one axis, 8.45mm offset on other

        // Parameters
        cylinder_radius = 23.5/2;
        cylinder_height = spacerh;
        hole_diameter = 4.826 + 0.4;
        hole_radius = hole_diameter / 2;
        hole_offset = 8.45;

        // Main cylinder
        difference() {
            // Base cylinder
            cylinder(h = cylinder_height, r = cylinder_radius, center = false);
            
            // Subtract the offset hole
            translate([hole_offset, 0, -0.1]) {
                cylinder(h = cylinder_height + 0.2, r = hole_radius, center = false);
            }
        }
}

module spacers() {
    // the pyrgeometer is 28.19mm base to the top of it sensor 
        
    // for the tops to be in same spot we have to add 29.92 - 28.19mm = 1.73mm spacer
    translate([0, 0, -60]) {
        spacer(29.92 - 28.19);
    }

    // make a 1.05mm one just to be safe too (its diagram height)

    translate([0, 0, -70]) {
        spacer(29.24 - 28.19);
    }
}

scale([1, 1, 1]) {
    housing();
    cap();
}
//spacers();
